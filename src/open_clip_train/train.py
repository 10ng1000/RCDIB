import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def save_tsne_visualization(model_out, tb_writer, step, checkpoint_path, sample_size=1000):
    """
    生成并保存t-SNE可视化图表到TensorBoard
    
    Args:
        model_out: 模型输出，包含各种特征
        tb_writer: TensorBoard writer
        step: 当前训练步数
        checkpoint_path: 检查点保存路径
        sample_size: 用于t-SNE的样本数量
    """
    try:
        # 设置图表样式
        plt.style.use('default')
        
        # 收集特征数据
        features_to_plot = {}
        
        # 原始特征
        if 'image_features' in model_out:
            image_features = model_out['image_features'].detach().cpu().numpy()
            features_to_plot['Original Image Features'] = image_features[:sample_size]
        
        if 'text_features' in model_out:
            text_features = model_out['text_features'].detach().cpu().numpy()
            features_to_plot['Original Text Features'] = text_features[:sample_size]
        
        # IBM压缩特征
        if 'image_mu' in model_out:
            image_mu = model_out['image_mu'].detach().cpu().numpy()
            features_to_plot['IBM Image Latent (μ)'] = image_mu[:sample_size]
        
        if 'text_mu' in model_out:
            text_mu = model_out['text_mu'].detach().cpu().numpy()
            features_to_plot['IBM Text Latent (μ)'] = text_mu[:sample_size]
        
        # 重构特征
        if 'reconstructed_image_features' in model_out:
            reconstructed_image = model_out['reconstructed_image_features'].detach().cpu().numpy()
            features_to_plot['Reconstructed Image Features'] = reconstructed_image[:sample_size]
        
        if 'reconstructed_text_features' in model_out:
            reconstructed_text = model_out['reconstructed_text_features'].detach().cpu().numpy()
            features_to_plot['Reconstructed Text Features'] = reconstructed_text[:sample_size]
        
        if not features_to_plot:
            return
        
        # 创建t-SNE可视化
        n_plots = len(features_to_plot)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # 定义颜色
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#F77F00']
        
        for idx, (feature_name, features) in enumerate(features_to_plot.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            try:
                # 如果特征维度太高，先用PCA降维
                if features.shape[1] > 50:
                    pca = PCA(n_components=50, random_state=42)
                    features_reduced = pca.fit_transform(features)
                else:
                    features_reduced = features
                
                # 应用t-SNE
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, features.shape[0]-1))
                features_2d = tsne.fit_transform(features_reduced)
                
                # 绘制散点图
                scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                                   c=colors[idx % len(colors)], alpha=0.6, s=20)
                
                ax.set_title(f'{feature_name}\n(Step {step})', fontweight='bold', fontsize=12)
                ax.set_xlabel('t-SNE Component 1')
                ax.set_ylabel('t-SNE Component 2')
                ax.grid(True, alpha=0.3)
                
                # 添加统计信息
                variance_explained = np.var(features_2d, axis=0).sum()
                ax.text(0.02, 0.98, f'Variance: {variance_explained:.2f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes,
                       ha='center', va='center', fontsize=10, color='red')
                ax.set_title(f'{feature_name} (Error)', fontweight='bold')
        
        # 隐藏多余的子图
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Feature Distribution Analysis - Training Step {step}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存到TensorBoard
        tb_writer.add_figure('IBM/Feature_Distribution_tSNE', fig, step)
        
        # 也保存到文件
        os.makedirs(os.path.join(checkpoint_path, 'tsne_plots'), exist_ok=True)
        fig.savefig(os.path.join(checkpoint_path, 'tsne_plots', f'tsne_step_{step}.png'), 
                   dpi=300, bbox_inches='tight')
        
        plt.close(fig)
        
        # 生成特征统计对比图
        create_feature_statistics_plot(features_to_plot, tb_writer, step, checkpoint_path)
        
    except Exception as e:
        logging.warning(f"Failed to generate t-SNE visualization: {e}")


def create_feature_statistics_plot(features_dict, tb_writer, step, checkpoint_path):
    """
    创建特征统计对比图
    """
    try:
        # 计算统计信息
        stats = {}
        for name, features in features_dict.items():
            stats[name] = {
                'mean': np.mean(features),
                'std': np.std(features),
                'min': np.min(features),
                'max': np.max(features),
                'l2_norm': np.linalg.norm(features, axis=1).mean()
            }
        
        # 创建统计对比图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        stat_names = ['mean', 'std', 'min', 'max', 'l2_norm']
        stat_titles = ['Mean Values', 'Standard Deviation', 'Minimum Values', 
                      'Maximum Values', 'L2 Norm']
        
        for i, (stat_name, title) in enumerate(zip(stat_names, stat_titles)):
            if i >= len(axes):
                break
                
            ax = axes[i]
            names = list(stats.keys())
            values = [stats[name][stat_name] for name in names]
            
            bars = ax.bar(range(len(names)), values, 
                         color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#F77F00'][:len(names)])
            
            ax.set_title(title, fontweight='bold')
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([name.replace(' ', '\n') for name in names], rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 隐藏最后一个子图
        axes[-1].set_visible(False)
        
        plt.suptitle(f'Feature Statistics Comparison - Step {step}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存到TensorBoard
        tb_writer.add_figure('IBM/Feature_Statistics', fig, step)
        
        # 保存到文件
        fig.savefig(os.path.join(checkpoint_path, 'tsne_plots', f'statistics_step_{step}.png'), 
                   dpi=300, bbox_inches='tight')
        
        plt.close(fig)
        
    except Exception as e:
        logging.warning(f"Failed to generate feature statistics plot: {e}")


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()      
    
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    model_out = {}
    
    # 冻结除了InformationBottleneckModule以外的参数
    if args.pretrain_ibm:
        # 在pretrain_ibm模式下，model和dist_model是同一个模型
        # 教师模型不学习
        for name,  param in model.named_parameters():
            if 'image_ibm' not in name and 'text_ibm' not in name:
                param.requires_grad = False
        # for name, param in model.named_parameters():
        #     if 'image_ibm' not in name and 'text_ibm' not in name:
        #         param.requires_grad = False
        #     else:
        #         param.requires_grad = True
                
        model.train()

    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        # Handle batch data - check if indices are included for teacher features
        if len(batch) == 3:  # images, texts, indices
            images, texts, batch_indices = batch
        else:  # images, texts
            images, texts = batch
            batch_indices = None

        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        
        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                
                if args.distill:
                    if not args.pretrain_ibm:
                        with torch.no_grad():
                            dist_model_out = dist_model(images, texts)
                        model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                    else:
                        # 在pretrain_ibm模式下，model和dist_model是同一个模型，不需要重复计算
                        # 直接使用model_out的结果作为dist_模型的输出
                        model_out.update({f'dist_{k}': v for k, v in model_out.items()})
                
                if args.pretrain_ibm:
                    model_out.update({
                        "pretrain_ibm": True,
                    })
                
                if args.enable_ibm_module:
                    # 如果启用了ibm模块，传入epoch信息
                    model_out.update({
                        "epoch": epoch,
                    })
                
                losses = loss(**model_out, output_dict=True)
                total_loss = sum(losses.values())
                losses["loss"] = total_loss
                

            backward(total_loss, scaler)
            
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: gradient norm = {param.grad.norm().item()}")
            #     else:
            #         print(f"{name}: no gradient")
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        if key in ['teacher_image_features', 'teacher_text_features']:
                            # For teacher features, use the cached values directly
                            inputs[key] = accumulated[j]
                        else:
                            inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                # f"Data (t): {data_time_m.avg:.3f} "
                # f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:7f} "
                # f"Logit Scale: {logit_scale_scalar:.3f} " 
                + loss_log
            )
            if args.enable_ibm_module:
                ibm_loss = losses.get("image_rec_loss", 0.0) + losses.get("text_rec_loss", 0.0) + losses.get("image_kl_loss", 0.0) + losses.get("text_kl_loss", 0.0)
                teacher_ibm_loss = losses.get("teacher_image_rec_loss", 0.0) + losses.get("teacher_text_rec_loss", 0.0) + losses.get("teacher_image_kl_loss", 0.0) + losses.get("teacher_text_kl_loss", 0.0)
                logging.info(f"IBM Loss: {ibm_loss:} " + f"Teacher IBM Loss: {teacher_ibm_loss:}")
                if not args.pretrain_ibm:
                    mmd_loss = losses.get("text_mmd_loss", 0.0) + losses.get("image_mmd_loss", 0.0)
                    dskl_loss = losses.get("text_dskl_loss", 0.0) + losses.get("image_dskl_loss", 0.0)
                    # feature_loss = losses.get("text_feature_loss", 0.0) + losses.get("image_feature_loss", 0.0)
                    logging.info(f"MMD Loss: {mmd_loss} " + f"DSKL Loss: {dskl_loss} ")
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                # "data_time": data_time_m.val,
                # "batch_time": batch_time_m.val,
                # "samples_per_second": samples_per_second,
                # "samples_per_second_per_gpu": samples_per_second_per_gpu,
                # "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})
            
            # Add IBM-specific metrics to tensorboard
            if args.enable_ibm_module:
                # Individual IBM loss components
                ibm_metrics = {}
                
                # Reconstruction losses
                image_rec_loss = losses.get("image_rec_loss", torch.tensor(0.0))
                text_rec_loss = losses.get("text_rec_loss", torch.tensor(0.0))
                ibm_metrics["image_rec_loss"] = image_rec_loss.item() if hasattr(image_rec_loss, 'item') else image_rec_loss
                ibm_metrics["text_rec_loss"] = text_rec_loss.item() if hasattr(text_rec_loss, 'item') else text_rec_loss
                
                # KL divergence losses (regularization)
                image_kl_loss = losses.get("image_kl_loss", torch.tensor(0.0))
                text_kl_loss = losses.get("text_kl_loss", torch.tensor(0.0))
                ibm_metrics["image_kl_loss"] = image_kl_loss.item() if hasattr(image_kl_loss, 'item') else image_kl_loss
                ibm_metrics["text_kl_loss"] = text_kl_loss.item() if hasattr(text_kl_loss, 'item') else text_kl_loss
                
                # Teacher model IBM losses (if not pretraining)
                if not args.pretrain_ibm:
                    teacher_image_rec_loss = losses.get("teacher_image_rec_loss", torch.tensor(0.0))
                    teacher_text_rec_loss = losses.get("teacher_text_rec_loss", torch.tensor(0.0))
                    teacher_image_kl_loss = losses.get("teacher_image_kl_loss", torch.tensor(0.0))
                    teacher_text_kl_loss = losses.get("teacher_text_kl_loss", torch.tensor(0.0))
                    
                    ibm_metrics["teacher_image_rec_loss"] = teacher_image_rec_loss.item() if hasattr(teacher_image_rec_loss, 'item') else teacher_image_rec_loss
                    ibm_metrics["teacher_text_rec_loss"] = teacher_text_rec_loss.item() if hasattr(teacher_text_rec_loss, 'item') else teacher_text_rec_loss
                    ibm_metrics["teacher_image_kl_loss"] = teacher_image_kl_loss.item() if hasattr(teacher_image_kl_loss, 'item') else teacher_image_kl_loss
                    ibm_metrics["teacher_text_kl_loss"] = teacher_text_kl_loss.item() if hasattr(teacher_text_kl_loss, 'item') else teacher_text_kl_loss
                    
                    # Distillation losses between student and teacher latent spaces
                    text_mmd_loss = losses.get("text_mmd_loss", torch.tensor(0.0))
                    image_mmd_loss = losses.get("image_mmd_loss", torch.tensor(0.0))
                    text_dskl_loss = losses.get("text_dskl_loss", torch.tensor(0.0))
                    image_dskl_loss = losses.get("image_dskl_loss", torch.tensor(0.0))
                    
                    ibm_metrics["text_mmd_loss"] = text_mmd_loss.item() if hasattr(text_mmd_loss, 'item') else text_mmd_loss
                    ibm_metrics["image_mmd_loss"] = image_mmd_loss.item() if hasattr(image_mmd_loss, 'item') else image_mmd_loss
                    ibm_metrics["text_dskl_loss"] = text_dskl_loss.item() if hasattr(text_dskl_loss, 'item') else text_dskl_loss
                    ibm_metrics["image_dskl_loss"] = image_dskl_loss.item() if hasattr(image_dskl_loss, 'item') else image_dskl_loss
                
                # Composite IBM metrics for analysis
                total_rec_loss = ibm_metrics["image_rec_loss"] + ibm_metrics["text_rec_loss"]
                total_kl_loss = ibm_metrics["image_kl_loss"] + ibm_metrics["text_kl_loss"]
                ibm_metrics["total_rec_loss"] = total_rec_loss
                ibm_metrics["total_kl_loss"] = total_kl_loss
                ibm_metrics["total_ibm_loss"] = total_rec_loss + total_kl_loss
                
                if not args.pretrain_ibm:
                    teacher_total_rec_loss = ibm_metrics["teacher_image_rec_loss"] + ibm_metrics["teacher_text_rec_loss"]
                    teacher_total_kl_loss = ibm_metrics["teacher_image_kl_loss"] + ibm_metrics["teacher_text_kl_loss"]
                    total_mmd_loss = ibm_metrics["text_mmd_loss"] + ibm_metrics["image_mmd_loss"]
                    total_dskl_loss = ibm_metrics["text_dskl_loss"] + ibm_metrics["image_dskl_loss"]
                    
                    ibm_metrics["teacher_total_rec_loss"] = teacher_total_rec_loss
                    ibm_metrics["teacher_total_kl_loss"] = teacher_total_kl_loss
                    ibm_metrics["teacher_total_ibm_loss"] = teacher_total_rec_loss + teacher_total_kl_loss
                    ibm_metrics["total_mmd_loss"] = total_mmd_loss
                    ibm_metrics["total_dskl_loss"] = total_dskl_loss
                    ibm_metrics["total_distill_loss"] = total_mmd_loss + total_dskl_loss
                
                # with torch.no_grad():
                #     # 检查特征距离
                #     text_z = model_out['text_z']
                #     image_z = model_out['image_z']
                #     dist_text_z = model_out.get('dist_text_z', None)
                #     dist_image_z = model_out.get('dist_image_z', None)
                #     text_dist = torch.norm(text_z - dist_text_z, dim=-1).mean()
                #     image_dist = torch.norm(image_z - dist_image_z, dim=-1).mean()
                    
                #     # 计算相对距离（归一化后的距离）
                #     text_rel_dist = text_dist / (torch.norm(text_z, dim=-1).mean() + 1e-8)
                #     image_rel_dist = image_dist / (torch.norm(image_z, dim=-1).mean() + 1e-8)
                    
                #     # 计算余弦相似度
                #     text_cos_sim = F.cosine_similarity(text_z, dist_text_z, dim=-1).mean()
                #     image_cos_sim = F.cosine_similarity(image_z, dist_image_z, dim=-1).mean()
                    
                #     logging.info(f"Text feature distance: {text_dist:.6f} (relative: {text_rel_dist:.6f})")
                #     logging.info(f"Image feature distance: {image_dist:.6f} (relative: {image_rel_dist:.6f})")
                #     logging.info(f"Text cosine similarity: {text_cos_sim:.6f}")
                #     logging.info(f"Image cosine similarity: {image_cos_sim:.6f}")
                #     logging.info(f"Text z norm: {torch.norm(text_z, dim=-1).mean():.6f}, Teacher: {torch.norm(dist_text_z, dim=-1).mean():.6f}")
                #     logging.info(f"Image z norm: {torch.norm(image_z, dim=-1).mean():.6f}, Teacher: {torch.norm(dist_image_z, dim=-1).mean():.6f}")
                
                # Add IBM metrics to log_data with proper naming
                for name, val in ibm_metrics.items():
                    log_data[f"train/ibm/{name}"] = val
            
            log_data = {"train/" + name if not name.startswith("train/") else name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
                
                # Add additional IBM analysis charts if enabled
                if args.enable_ibm_module and 'image_mu' in model_out and 'image_log_var' in model_out:
                    # Log latent space statistics (if available in model output)
                    image_mu = model_out['image_mu']
                    image_log_var = model_out['image_log_var']
                    text_mu = model_out['text_mu']
                    text_log_var = model_out['text_log_var']
                    
                    # Latent space statistics
                    tb_writer.add_scalar('train/ibm/latent/image_mu_mean', image_mu.mean().item(), step)
                    tb_writer.add_scalar('train/ibm/latent/image_mu_std', image_mu.std().item(), step)
                    tb_writer.add_scalar('train/ibm/latent/image_sigma_mean', (0.5 * image_log_var).exp().mean().item(), step)
                    tb_writer.add_scalar('train/ibm/latent/image_sigma_std', (0.5 * image_log_var).exp().std().item(), step)
                    
                    tb_writer.add_scalar('train/ibm/latent/text_mu_mean', text_mu.mean().item(), step)
                    tb_writer.add_scalar('train/ibm/latent/text_mu_std', text_mu.std().item(), step)
                    tb_writer.add_scalar('train/ibm/latent/text_sigma_mean', (0.5 * text_log_var).exp().mean().item(), step)
                    tb_writer.add_scalar('train/ibm/latent/text_sigma_std', (0.5 * text_log_var).exp().std().item(), step)
                    
                    # Information bottleneck metrics
                    if 'image_features' in model_out and 'text_features' in model_out:
                        image_compression_ratio = image_mu.shape[-1] / model_out['image_features'].shape[-1]
                        text_compression_ratio = text_mu.shape[-1] / model_out['text_features'].shape[-1]
                        tb_writer.add_scalar('train/ibm/compression/image_ratio', image_compression_ratio, step)
                        tb_writer.add_scalar('train/ibm/compression/text_ratio', text_compression_ratio, step)
                        
                        # Feature similarity between original and reconstructed
                        if 'reconstructed_image_features' in model_out and 'reconstructed_text_features' in model_out:
                            image_cosine_sim = F.cosine_similarity(
                                model_out['image_features'], 
                                model_out['reconstructed_image_features'], 
                                dim=-1
                            ).mean()
                            text_cosine_sim = F.cosine_similarity(
                                model_out['text_features'], 
                                model_out['reconstructed_text_features'], 
                                dim=-1
                            ).mean()
                            tb_writer.add_scalar('train/ibm/similarity/image_cosine', image_cosine_sim.item(), step)
                            tb_writer.add_scalar('train/ibm/similarity/text_cosine', text_cosine_sim.item(), step)
                
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
