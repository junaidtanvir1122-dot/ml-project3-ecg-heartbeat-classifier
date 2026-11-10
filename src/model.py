import torch
import torch.nn as nn

class ECGNet(nn.Module):
    def __init__(self, sizes=(187, 128, 64, 5)):
        super().__init__()
        torch.manual_seed(0)                                   # har dafa wohi shuruaati weights
        self.W = nn.ParameterList([nn.Parameter(torch.randn(n_out, n_in) * (2 / n_in) ** 0.5)
                                   for n_in, n_out in zip(sizes[:-1], sizes[1:])])  # He init
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(n_out)) for n_out in sizes[1:]])

    def forward(self, x):                                      # x: (B, 187)
        for W, b in zip(self.W[:-1], self.b[:-1]):
            x = torch.clamp(x @ W.T + b, min=0)                # hidden layers: Z = A Wᵀ + b, phir ReLU
        return x @ self.W[-1].T + self.b[-1]                   # output layer: sirf logits, ReLU nahi

def cross_entropy(logits, y):
    # L = -z_y + log(sum e^z), phir batch ka mean
    return (torch.logsumexp(logits, dim=1) - logits[torch.arange(len(y)), y]).mean()

if __name__ == "__main__":
    model = ECGNet()
    print("trainable parameters:", sum(p.numel() for p in model.parameters()))
    print("shapes:", [tuple(p.shape) for p in model.parameters()])
    print("logits shape:", model(torch.randn(4, 187)).shape)