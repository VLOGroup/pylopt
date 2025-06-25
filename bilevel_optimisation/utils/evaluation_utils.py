import torch

def compute_psnr(y_true: torch.Tensor, y_pred: torch.Tensor, max_pix_value: float = 1.0) -> torch.Tensor:
    mse = torch.mean(((y_true - y_pred) ** 2), dim=(-2, -1))
    return 20 * torch.log10(max_pix_value / torch.sqrt(mse))
