import torch
from torch.nn.functional import mse_loss


@torch.no_grad()
def compute_recon_loss(model, loader):
    torch._C._mps_emptyCache()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    y_true = torch.Tensor()
    y_pred = torch.Tensor()

    model.to(device)
    for X, _ in loader:
        batch_gpu = X.to(device)
        y_true = torch.cat((y_true, X))
        recon = model.generate(batch_gpu).detach().cpu()
        y_pred = torch.cat((y_pred, recon))

    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    recon_loss = mse_loss(y_pred, y_true)
    return recon_loss
