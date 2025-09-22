from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from .chebyKANLayer import ChebyKANLinear
import logging


try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        if logit_bias is not None:
            logits_per_image += logit_bias
            logits_per_text += logit_bias

        return logits_per_image, logits_per_text

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            logit_bias=None,
            output_dict=False,
    ):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

def compute_rbf_kernel(x, y, sigma=1.0):
    """
    Computes the RBF kernel matrix between two sets of embeddings x and y.

    Args:
        x: A tensor of shape (batch_size_1, embedding_dim).
        y: A tensor of shape (batch_size_2, embedding_dim).
        sigma: The bandwidth of the RBF kernel.

    Returns:
        A tensor of shape (batch_size_1, batch_size_2) containing the RBF kernel matrix.
    """

    # Compute the squared Euclidean distance matrix between x and y
    dist = torch.sum(x ** 2, dim=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
    # Compute the RBF kernel matrix
    kernel = torch.exp(-dist / (2 * sigma ** 2))

    return kernel

def MMD_loss(x, y, sigma=1.0):
    n = x.size(0)
    m = y.size(0)
    # Compute the RBF kernel matrix for x and y
    Kxx = compute_rbf_kernel(x, x, sigma)
    Kxy = compute_rbf_kernel(x, y, sigma)
    Kyy = compute_rbf_kernel(y, y, sigma)

    loss = torch.sum(Kxx) + torch.sum(Kyy) - 2 * torch.sum(Kxy)

    return loss/(n*m)

# def MMD_loss(x, y, sigma=1.0):
#     """
#     Computes the Maximum Mean Discrepancy (MMD) loss between two distributions.
    
#     Args:
#         x: A tensor of shape (n, embedding_dim) - samples from first distribution
#         y: A tensor of shape (m, embedding_dim) - samples from second distribution
#         sigma: The bandwidth of the RBF kernel
    
#     Returns:
#         MMD loss value
#     """
#     # Get batch sizes
#     n = x.size(0)
#     m = y.size(0)
#     # Compute kernel matrices
#     Kxx = compute_rbf_kernel(x, x, sigma)
#     Kyy = compute_rbf_kernel(y, y, sigma)
#     Kxy = compute_rbf_kernel(x, y, sigma)
    
#     # Remove diagonal elements for Kxx and Kyy (unbiased estimator)
#     Kxx_sum = torch.sum(Kxx) - torch.sum(torch.diag(Kxx))
#     Kyy_sum = torch.sum(Kyy) - torch.sum(torch.diag(Kyy))
#     Kxy_sum = torch.sum(Kxy)
    
#     # Unbiased MMD² estimator
#     mmd_loss = Kxx_sum / (n * (n - 1)) + Kyy_sum / (m * (m - 1)) - 2 * Kxy_sum / (n * m)
    
#     return mmd_loss

def gaussian_kl_divergence(mu1, logvar1, mu2, logvar2):
    """
    Calculate the KL divergence KL(N(mu1, var1) || N(mu2, var2))
    
    Args:
        mu1: tensor, mean of the first Gaussian distribution
        logvar1: tensor, log variance of the first Gaussian distribution  
        mu2: tensor, mean of the second Gaussian distribution
        logvar2: tensor, log variance of the second Gaussian distribution

    Returns:
        kl_divergence: tensor, KL divergence between two Gaussian distributions
    """
    # For multivariate Gaussian with diagonal covariance:
    # KL(N(μ1,Σ1) || N(μ2,Σ2)) = 0.5 * [tr(Σ2^-1 Σ1) + (μ2-μ1)^T Σ2^-1 (μ2-μ1) + log(|Σ2|/|Σ1|) - k]
    
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    
    # Calculate each term
    trace_term = torch.sum(var1 / var2, dim=-1)  # tr(Σ2^-1 Σ1)
    mahalanobis_term = torch.sum((mu2 - mu1).pow(2) / var2, dim=-1)  # (μ2-μ1)^T Σ2^-1 (μ2-μ1)
    log_det_term = torch.sum(logvar2 - logvar1, dim=-1)  # log|Σ2| - log|Σ1|
    constant_term = mu1.shape[-1]  # dimension k
    
    kl_divergence = 0.5 * (trace_term + mahalanobis_term + log_det_term - constant_term)
    
    return torch.mean(kl_divergence)

# def gaussian_kl_divergence(mu1, logvar1, mu2, logvar2):
#     """
#     Calculate the KL divergence between two Gaussian distributions

#     Args:
#         mu1: tensor, mean of the first Gaussian distribution
#         logvar1: tensor, log variance of the first Gaussian distribution
#         mu2: tensor, mean of the second Gaussian distribution
#         logvar2: tensor, log variance of the second Gaussian distribution

#     Returns:
#         kl_divergence: tensor, KL divergence between two Gaussian distributions
#     """
#     # Calculate the diagonal elements of covariance matrix
#     var1 = torch.exp(logvar1)
#     var2 = torch.exp(logvar2)

#     # Calculate the KL divergence
#     kl_divergence = 0.5 * (torch.sum(var1 / var2, dim=-1)
#                            + torch.sum((mu2 - mu1).pow(2) / var2, dim=-1)
#                            + torch.sum(logvar2, dim=-1)
#                            - torch.sum(logvar1, dim=-1)
#                            - mu1.shape[-1])

#     return torch.sum(kl_divergence)/(mu1.shape[0]*mu1.shape[1])

class ClipWithIBMLoss(ClipLoss):
    def __init__(
            self,
            embed_dim: int = 512,
            distill_embed_dim: int = 768,
            kl_weight: float = 1,
            local_loss: bool = False,
            gather_with_grad: bool = False,
            cache_labels: bool = False,
            rank: int = 0,
            world_size: int = 1,
            use_horovod: bool = False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod,
        )
        self.kl_weight = kl_weight
        self.embed_dim = embed_dim
        self.distill_embed_dim = distill_embed_dim
        
        # For adaptive weights with EMA smoothing
        self.register_buffer('ema_mmd_weight', torch.tensor(1.0))
        self.register_buffer('ema_dskl_weight', torch.tensor(1.0))
        self.ema_momentum = 0.99  # EMA momentum for smoothing
        # feature distill
        # if distill_embed_dim != embed_dim:
        #     self.visual_proj = ChebyKANLinear(embed_dim, distill_embed_dim, degree=3)
        #     self.text_proj = ChebyKANLinear(embed_dim, distill_embed_dim, degree=3)

        
    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)
    
    def forward(
            self,
            image_features: torch.Tensor,
            text_features: torch.Tensor,
            logit_scale: torch.Tensor,
            reconstructed_image_features: Optional[torch.Tensor],
            reconstructed_text_features: Optional[torch.Tensor],
            epoch: int = 0,
            pretrain_ibm: bool = False,
            image_z: Optional[torch.Tensor] = None,
            text_z: Optional[torch.Tensor] = None,
            image_mu: Optional[torch.Tensor] = None,
            image_log_var: Optional[torch.Tensor] = None,
            text_mu: Optional[torch.Tensor] = None,
            text_log_var: Optional[torch.Tensor] = None,
            dist_image_features: Optional[torch.Tensor] = None,
            dist_text_features: Optional[torch.Tensor] = None,
            dist_reconstructed_image_features: Optional[torch.Tensor] = None,
            dist_reconstructed_text_features: Optional[torch.Tensor] = None,
            logit_bias: Optional[torch.Tensor] = None,
            dist_image_z: Optional[torch.Tensor] = None,
            dist_text_z: Optional[torch.Tensor] = None,
            dist_image_mu: Optional[torch.Tensor] = None,
            dist_image_log_var: Optional[torch.Tensor] = None,
            dist_text_log_var: Optional[torch.Tensor] = None,
            dist_text_mu: Optional[torch.Tensor] = None,
            dist_logit_scale: Optional[torch.Tensor] = None,
            output_dict: bool = False,
    ):
        
        losses_dict = {}
        ablation_MMD = False
        MMD_weight = self.kl_weight
        
        if not pretrain_ibm:
            #正常蒸馏模式：计算CLIP对比损失和蒸馏损失
            #不带原来clip损失
            # clip_loss = super().forward(
            #     image_features, text_features, logit_scale, logit_bias, output_dict=False
            # )
            # losses_dict["contrastive_loss"] = clip_loss

            #contrastive蒸馏损失
            logits_per_image, logits_per_text = self.get_logits(
                image_features, text_features, logit_scale
            )
            dist_logits_per_image, dist_logits_per_text = self.get_logits(
                dist_image_features, dist_text_features, dist_logit_scale
            )
            
            distill_loss = (
                self.dist_loss(dist_logits_per_image, logits_per_image) +
                self.dist_loss(dist_logits_per_text, logits_per_text)
            ) / 2
            
            losses_dict["distill_loss"] = distill_loss
            
            # 两者的重构损失
            image_rec_loss = self.kl_weight * F.mse_loss(reconstructed_image_features, image_features, reduction='mean')
            text_rec_loss = self.kl_weight * F.mse_loss(reconstructed_text_features, text_features, reduction='mean')
            losses_dict["image_rec_loss"] = image_rec_loss
            losses_dict["text_rec_loss"] = text_rec_loss
            
            image_kl_loss = self.kl_weight * 0.5 * torch.sum(image_mu.pow(2) + image_log_var.exp() - 1.0 - image_log_var) / image_mu.shape[0]
            text_kl_loss = self.kl_weight * 0.5 * torch.sum(text_mu.pow(2) + text_log_var.exp() - 1.0 - text_log_var) / text_mu.shape[0]
            losses_dict["image_kl_loss"] = image_kl_loss
            losses_dict["text_kl_loss"] = text_kl_loss
            
            # 教师模型的重构和KL损失
            teacher_image_rec_loss = self.kl_weight * F.mse_loss(dist_reconstructed_image_features, dist_image_features, reduction='mean')
            teacher_text_rec_loss = self.kl_weight * F.mse_loss(dist_reconstructed_text_features, dist_text_features, reduction='mean')
            losses_dict["teacher_image_rec_loss"] = teacher_image_rec_loss
            losses_dict["teacher_text_rec_loss"] = teacher_text_rec_loss
            
            teacher_image_kl_loss = self.kl_weight * 0.5 * torch.sum(dist_image_mu.pow(2) + dist_image_log_var.exp() - 1.0 - dist_image_log_var) / dist_image_mu.shape[0]
            teacher_text_kl_loss = self.kl_weight * 0.5 * torch.sum(dist_text_mu.pow(2) + dist_text_log_var.exp() - 1.0 - dist_text_log_var) / dist_text_mu.shape[0]
            losses_dict["teacher_image_kl_loss"] = teacher_image_kl_loss
            losses_dict["teacher_text_kl_loss"] = teacher_text_kl_loss
            
            # 根据训练阶段调整MMD权重，考虑到数量级平衡
            # if epoch < 1 or ablation_MMD:
            #     MMD_weight = 0.0
            # 学生和教师之间的蒸馏损失
            batch_size = image_features.shape[0]
            
            # Calculate base MMD and DSKL losses without scaling
            # base_text_mmd_loss = MMD_loss(text_mu, dist_text_mu, batch_size) + 0.3 * MMD_loss(text_log_var, dist_text_log_var, batch_size)
            # base_image_mmd_loss = MMD_loss(image_mu, dist_image_mu, batch_size) + 0.3 * MMD_loss(image_log_var, dist_image_log_var, batch_size)
            # 对比z
            # text_z_norm = F.normalize(text_z, dim=-1)
            # dist_text_z_norm = F.normalize(dist_text_z, dim=-1)
            # image_z_norm = F.normalize(image_z, dim=-1)
            # dist_image_z_norm = F.normalize(dist_image_z, dim=-1)
            
            base_text_mmd_loss = MMD_loss(text_z, dist_text_z)
            base_image_mmd_loss = MMD_loss(image_z, dist_image_z)

            base_text_dskl_loss = gaussian_kl_divergence(
                text_mu, text_log_var, dist_text_mu, dist_text_log_var
            )
            base_image_dskl_loss = gaussian_kl_divergence(
                image_mu, image_log_var, dist_image_mu, dist_image_log_var
            )
            # 之前是10 100. 1e-2 10不行
            # 调整权重以匹配其他损失的数量级
            text_mmd_loss =  1e-2 * MMD_weight * base_text_mmd_loss
            image_mmd_loss =  1e-2 * MMD_weight * base_image_mmd_loss

            text_dskl_loss = 10 * MMD_weight * base_text_dskl_loss
            image_dskl_loss = 10 * MMD_weight * base_image_dskl_loss
            losses_dict["text_mmd_loss"] = text_mmd_loss
            losses_dict["image_mmd_loss"] = image_mmd_loss
            losses_dict["text_dskl_loss"] = text_dskl_loss
            losses_dict["image_dskl_loss"] = image_dskl_loss
        # total_loss += 0.01 * self.kl_weight * (text_mmd_loss / (text_mmd_loss.detach() + 1e-6) + image_mmd_loss / (image_mmd_loss.detach() + 1e-6) + text_dskl_loss / (text_dskl_loss.detach() + 1e-6) + image_dskl_loss / (image_dskl_loss.detach() + 1e-6))
        
        else:
            # clip_loss = super().forward(
            #     image_features, text_features, logit_scale, logit_bias, output_dict=False
            # )
            # losses_dict["contrastive_loss"] = clip_loss
            # pretrain_ibm模式：只训练IBM模块，不计算学生教师之间的蒸馏损失
            # 在这种模式下，image_features等同于dist_image_features（都来自同一个模型）
            # 只计算重构和KL损失，用于训练IBM模块
            image_rec_loss = self.kl_weight * F.mse_loss(reconstructed_image_features, image_features, reduction='mean')
            text_rec_loss = self.kl_weight * F.mse_loss(reconstructed_text_features, text_features, reduction='mean')
            losses_dict["image_rec_loss"] = image_rec_loss
            losses_dict["text_rec_loss"] = text_rec_loss

            image_kl_loss = self.kl_weight * 0.5 * torch.sum(image_mu.pow(2) + image_log_var.exp() - 1.0 - image_log_var) / image_mu.shape[0]
            text_kl_loss = self.kl_weight * 0.5 * torch.sum(text_mu.pow(2) + text_log_var.exp() - 1.0 - text_log_var) / text_mu.shape[0]
            losses_dict["image_kl_loss"] = image_kl_loss
            losses_dict["text_kl_loss"] = text_kl_loss

        if output_dict:
            return losses_dict
        if pretrain_ibm:
            # 在pretrain_ibm模式下，只返回当前模型的IBM相关损失
            return losses_dict["image_kl_loss"], losses_dict["text_kl_loss"], losses_dict["image_rec_loss"], losses_dict["text_rec_loss"]
        return image_rec_loss, text_rec_loss, image_kl_loss, text_kl_loss, teacher_image_kl_loss, teacher_text_kl_loss, text_mmd_loss, image_mmd_loss, text_dskl_loss, image_dskl_loss, teacher_image_rec_loss, teacher_text_rec_loss

class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss
        else:
            clip_loss = torch.tensor(0, device=logits.device)

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def __init__(
            self,
            embed_dim: int = 512,
            distill_embed_dim: int = 768,
            local_loss: bool = False,
            gather_with_grad: bool = False,
            cache_labels: bool = False,
            rank: int = 0,
            world_size: int = 1,
            use_horovod: bool = False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod,
        )
        self.embed_dim = embed_dim
        self.distill_embed_dim = distill_embed_dim
        # feature distill
        # if distill_embed_dim != embed_dim:
        #     self.visual_proj = ChebyKANLinear(embed_dim, distill_embed_dim, degree=3)
        #     self.text_proj = ChebyKANLinear(embed_dim, distill_embed_dim, degree=3)

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        # contrastive_loss = (
        #     F.cross_entropy(logits_per_image, labels) +
        #     F.cross_entropy(logits_per_text, labels)
        # ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        # if self.embed_dim != self.distill_embed_dim:
        #     projected_image_features = self.visual_proj(image_features)
        #     projected_text_features = self.text_proj(text_features)
        #     normalized_image_features = F.normalize(projected_image_features, dim=-1)
        #     normalized_text_features = F.normalize(projected_text_features, dim=-1)

        # image_feature_loss = F.mse_loss(normalized_image_features, dist_image_features, normalized_image_features.shape[0])
        # text_feature_loss = F.mse_loss(normalized_text_features, dist_text_features, normalized_text_features.shape[0])

        # if output_dict:
        #     return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        # return contrastive_loss, distill_loss
        if output_dict:
            return {"distill_loss": distill_loss}
        return distill_loss



def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels: bool = False,
            rank: int = 0,
            world_size: int = 1,
            dist_impl: Optional[str] = None,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or 'bidir'  # default to bidir exchange for now, this will likely change
        assert self.dist_impl in ('bidir', 'shift', 'reduce', 'gather')

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            if self.dist_impl == 'bidir':
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right
                    )
                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "shift":
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right,
                    )
                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left
            elif self.dist_impl == "reduce":
                for i in range(self.world_size):
                    text_from_other = torch.distributed.nn.all_reduce(
                        text_features * (self.rank == i),
                        torch.distributed.ReduceOp.SUM,
                    )
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        text_from_other,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "gather":
                all_text = torch.distributed.nn.all_gather(text_features)
                for i in range(self.world_size):
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        all_text[i],
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                assert False

        return {"contrastive_loss": loss} if output_dict else loss
