import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as func


class CombDiceCross(_Loss):
    """
    The Sørensen-Dice Loss.
    """
    def __init__(self):
        super(CombDiceCross, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, wht: torch.tensor):
        """
        Computes the CrossEntropy and Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            inputs (:obj:`torch.Tensor`) : A tensor of shape (B, C, ..). The model prediction on which the loss has to
             be computed.
            targets (:obj:`torch.Tensor`) : A tensor of shape (B, C, ..). The ground truth.
        Returns:
            :obj:`torch.Tensor`: The Sørensen–Dice loss for each class or reduced according to reduction method.
        """
        if not inputs.size() == targets.size():
            raise ValueError("'Inputs' and 'Targets' must have the same shape.")

        eps = 0.000000001

        intersection = (func.softmax(inputs, dim=1) * targets).sum(0)
        union = (func.softmax(inputs, dim=1) + targets).sum(0)
        numerator = 2 * intersection
        denominator = union + eps

        loss_dic = (wht * (1 - (numerator / denominator))).sum()

        loss_cen = func.cross_entropy(inputs, torch.max(targets, 1)[1], weight=wht)

        return (loss_dic + loss_cen) / 2
