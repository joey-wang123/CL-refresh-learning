import torch.nn as nn
import torch.nn.functional as F

class CPR(nn.Module):
    def __init__(self):
        super(CPR, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1)
        
        return b.mean()