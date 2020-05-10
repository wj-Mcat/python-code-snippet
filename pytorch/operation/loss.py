import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        pass

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert source.dim() == 2
        assert target.dim() == 1

        score = -1. * source.log_softmax(dim=-1).gather(1, target.unsqueeze(1))
        return score


class BatchCrossEntropyLoss(nn.Module):
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source, target = source.clone(), target.clone()
        if source.dim() == 2:
            source.unsqueeze_(1)
            target.unsqueeze_(1)
        score = -1. * source.log_softmax(dim=-1).gather(-1, target.unsqueeze(-1))
        return score.sum()