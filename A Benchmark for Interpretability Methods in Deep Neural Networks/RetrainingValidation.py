"""
This tests the "validating the behavior or ROAR on artificial data part,
where I'm mostly concerned of the performance degradation of the method w/o retraining.
"""
import torch
import torch.nn as nn


class RandomModel(nn.Module):
    """
    In the original paper, it writes:
        \pmb{\text{x}}=\frac{\pmb{a}z}{10}+\pmb{d}\eta+\frac{\epsilon}{10},
        where it says only the first 4 elements in \pmb{a} is non zero and label y is determined by the sign of z,
        which is simply not possible. Hence we make the assumption that we should swap x and z
    """

    def __init__(self):
        super(RandomModel, self).__init__()
        self.a = torch.randn(1, 16)
        self.a[0, 4:] = torch.zeros(12)
        self.d = torch.randn(1, 16)

    def forward(self, x):
        batch_size = x.shape[0]
        return torch.sign(self.a * x / 10 + self.d * torch.randn(batch_size, 1) + torch.randn(batch_size, 1) / 10)


if __name__ == "__main__":
    rm = RandomModel()
    rm(torch.randn(100, 16))
