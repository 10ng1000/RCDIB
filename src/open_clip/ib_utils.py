import torch
import torch.nn as nn
import torch.nn.functional as F
from chebyKANLayer import ChebyKANLinear

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

def MMD_loss(x, y, sigma=1.0, args=None):
    # Compute the RBF kernel matrix for x and y
    Kxx = compute_rbf_kernel(x, x, sigma)
    Kxy = compute_rbf_kernel(x, y, sigma)
    Kyy = compute_rbf_kernel(y, y, sigma)

    loss = torch.sum(Kxx) + torch.sum(Kyy) - 2 * torch.sum(Kxy)

    return loss/(args.batch_size*args.batch_size)

def gaussian_kl_divergence(mu1, logvar1, mu2, logvar2):
    """
    Calculate the KL divergence between two Gaussian distributions

    Args:
        mu1: tensor, mean of the first Gaussian distribution
        logvar1: tensor, log variance of the first Gaussian distribution
        mu2: tensor, mean of the second Gaussian distribution
        logvar2: tensor, log variance of the second Gaussian distribution

    Returns:
        kl_divergence: tensor, KL divergence between two Gaussian distributions
    """
    # Calculate the diagonal elements of covariance matrix
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)

    # Calculate the KL divergence
    kl_divergence = 0.5 * (torch.sum(var1 / var2, dim=-1)
                           + torch.sum((mu2 - mu1).pow(2) / var2, dim=-1)
                           + torch.sum(logvar2, dim=-1)
                           - torch.sum(logvar1, dim=-1)
                           - mu1.shape[-1])

    return torch.sum(kl_divergence)/(mu1.shape[0]*mu1.shape[1])

class IBModel(nn.Module):

    def __init__(self, x,
                 n_input, n_hidden, n_z, pretrain):
        super(IBModel, self).__init__()

        # encoder
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(n_input[0], 32, (4, 4), stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, (4, 4), stride=2, padding=1),
        #     nn.ReLU(),

        # )
        self.x = x
        self.encoder = ChebyKANLinear(n_input, n_hidden, degree=3, drop_rate=0.0, drop_scale=False)
        
        self.latentmu = ChebyKANLinear(n_hidden, n_z, degree=3, drop_rate=0.0, drop_scale=False)
        self.latentep = ChebyKANLinear(n_hidden, n_z, degree=3, drop_rate=0.0, drop_scale=False)
        
        self.delatent = ChebyKANLinear(n_z, n_hidden, degree=3, drop_rate=0.0, drop_scale=False)
        self.decoder = ChebyKANLinear(n_hidden, n_input, degree=3, drop_rate=0.0, drop_scale=False)
        
        # for m in self.encoder:
        #     if isinstance(m, torch.nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        #         torch.nn.init.constant_(m.bias, 0)

        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 32, (4, 4), stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, n_input[0], (4, 4), stride=2, padding=1),
        #     nn.Sigmoid()
        # )

        # for m in self.decoder:
        #     if isinstance(m, torch.nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        #         torch.nn.init.constant_(m.bias, 0)


        # for m in self.decoder:
        #     if isinstance(m, torch.nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        #         torch.nn.init.constant_(m.bias, 0)

        self.pretrain = pretrain

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        z = mu + eps * std
        return z


    def forward(self, x):
        encoded = self.encoder(x)
        mu =self.latentmu(encoded)
        log = self.latentep(encoded)
        z = self.reparameterize(mu, log)
        #y=self.classifier(z)

        # z = torch.nn.functional.normalize(z, p=1.0, dim=1)

        x_bar=self.decoder(self.delatent(z))
        #q = self.clusteringLayer(z)

        return x_bar, z, mu, log
    
