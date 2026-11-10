import torch
from prepare import prepare
from model import ECGNet, cross_entropy

Xtr, ytr, *_ = prepare()
X, y = Xtr[:64], ytr[:64]                       # pehli 64 training rows
model = ECGNet()
W, b = model.W, model.b

loss = cross_entropy(model(X), y)
print(f"initial loss: {loss.item():.4f}  (log 5 = 1.6094)")
loss.backward()                                  # PyTorch ke gradients W[i].grad mein aa gaye

with torch.no_grad():                            # ab wohi gradient haath se nikalte hain
    A2 = torch.clamp(torch.clamp(X @ W[0].T + b[0], min=0) @ W[1].T + b[1], min=0)
    Yhat = torch.softmax(A2 @ W[2].T + b[2], dim=1)
    Y = torch.zeros(64, 5)
    Y[torch.arange(64), y] = 1                   # one-hot labels
    dW3 = (Yhat - Y).T @ A2 / 64
    db3 = (Yhat - Y).mean(dim=0)
    print("max diff W3:", (dW3 - W[2].grad).abs().max().item())
    print("max diff b3:", (db3 - b[2].grad).abs().max().item())