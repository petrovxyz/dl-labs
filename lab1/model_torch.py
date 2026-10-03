import torch
from torch import nn
import torch.nn.functional as F

def torch_reference(X, y, W1, b1, W2, b2):
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    ).double()

    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(W1.T))
        model[0].bias.copy_(torch.from_numpy(b1))

        model[2].weight.copy_(torch.from_numpy(W2.T))
        model[2].bias.copy_(torch.from_numpy(b2))

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).long()

    model.zero_grad()

    logits_t = model(X_t)

    loss_t = F.cross_entropy(
        logits_t,
        y_t,
        reduction="mean",
    )

    loss_t.backward()

    dW1_t = model[0].weight.grad.T.detach().numpy()
    db1_t = model[0].bias.grad.detach().numpy()

    dW2_t = model[2].weight.grad.T.detach().numpy()
    db2_t = model[2].bias.grad.detach().numpy()

    return (
        loss_t.item(),
        dW1_t,
        db1_t,
        dW2_t,
        db2_t,
    )