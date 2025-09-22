import torch
import torch.nn as nn

class ChebyKANLinear(nn.Module):
    def __init__(self, input_dim, output_dim, degree = 3, drop_rate = None, drop_scale = None):
        super(ChebyKANLinear, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        # self.drop_rate = drop_rate
        # self.drop_scale = drop_scale

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        self.epsilon = 1e-7
        self.pre_mul = False
        self.post_mul = False
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1/(input_dim*(degree+1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        b, c_in = x.shape
        if self.pre_mul:
            mul_1 = x[:, ::2]
            mul_2 = x[:, 1::2]
            mul_res = mul_1 * mul_2
            x = torch.cat([x[:, :x.shape[1]//2], mul_res], dim=1)
            
        x = x.view((b, c_in, 1)).expand(-1, -1, self.degree+1)
        x = torch.tanh(x)
        x = torch.tanh(x)
        x = torch.acos(x)
        x = x * self.arange
        x = x.cos()
        
        y = torch.einsum("bid,iod->bo", x, self.cheby_coeffs).view(-1, self.outdim)
        
        if self.post_mul:
            mul_1 = y[:, ::2]
            mul_2 = y[:, 1::2]
            mul_res = mul_1 * mul_2
            y = torch.cat([y[:, :y.shape[1]//2], mul_res], dim=1)

        # if self.training and self.drop_rate > 0:
        #     mask = torch.empty_like(y).bernoulli_(1 - self.drop_rate)
        #     if self.drop_scale:
        #         y = y * mask / (1 - self.drop_rate)  # 缩放
        #     else:
        #         y = y * mask

        return y