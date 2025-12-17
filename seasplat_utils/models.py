import torch
import torch.nn as nn

class BackscatterNetV2(nn.Module):
    '''
    估计后向散射 (Backscatter): B = B_inf * (1 - exp(-beta_b * z))
    '''
    def __init__(self, scale: float = 5.0):
        super().__init__()
        self.scale = scale
        # [关键修改] 初始化为 -5.0，经过 sigmoid 后接近 0，初始状态假设“几乎无散射”
        self.backscatter_conv_params = nn.Parameter(torch.ones(3, 1, 1, 1) * 0.0)
        self.B_inf = nn.Parameter(torch.rand(3, 1, 1))
        self.relu = nn.ReLU()

    def forward(self, depth):
        beta_b_conv = self.relu(torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.backscatter_conv_params)))
        backscatter = torch.sigmoid(self.B_inf) * (1 - torch.exp(-beta_b_conv))
        return backscatter

class AttenuateNetV3(nn.Module):
    '''
    估计衰减 (Attenuation): A = exp(-beta_d * z)
    '''
    def __init__(self, scale: float = 5.0):
        super().__init__()
        # [关键修改] 初始化为 -5.0，使初始衰减系数接近 0，透过率 exp(-0) 接近 1 (透明/无衰减)
        self.attenuation_conv_params = nn.Parameter(torch.ones(3, 1, 1, 1) * 0.0)
        self.scale = scale
        self.relu = nn.ReLU()

    def forward(self, depth):
        beta_d_conv = self.relu(torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.attenuation_conv_params)))
        attenuation_map = torch.exp(-beta_d_conv)
        return attenuation_map