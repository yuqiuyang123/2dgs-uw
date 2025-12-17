import torch
import torch.nn as nn

class BackscatterLoss(nn.Module):
    def __init__(self, cost_ratio=1000.):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.2)
        self.relu = nn.ReLU()
        self.cost_ratio = cost_ratio

    def forward(self, direct_signal):
        neg = self.smooth_l1(self.relu(-direct_signal), torch.zeros_like(direct_signal))
        pos = self.l1(self.relu(direct_signal), torch.zeros_like(direct_signal))
        return self.cost_ratio * neg + pos

class GrayWorldLoss(nn.Module):
    def __init__(self, target=0.5):
        super().__init__()
        self.target = target

    def forward(self, restored_image):
        means = torch.mean(restored_image, dim=[2, 3])
        return ((means - self.target) ** 2).mean()

class SaturationLoss(nn.Module):
    def __init__(self, thresh=0.7):
        super().__init__()
        self.thresh = thresh
        self.relu = nn.ReLU()

    def forward(self, restored_image):
        return (self.relu(restored_image - 1.0) + self.relu(-restored_image)).mean()

class AlphaBackgroundLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super().__init__()
        self.threshold = threshold
        self.l1 = nn.L1Loss()

    def forward(self, rgb_render, bg_color, alpha):
        # bg_color [3, 1, 1] vs rgb_render [3, H, W]
        diff = torch.norm(rgb_render - bg_color, dim=0, keepdim=True)
        mask = diff < self.threshold
        if mask.sum() > 0:
            return self.l1(alpha[mask], torch.zeros_like(alpha[mask]))
        return torch.tensor(0.0).cuda()