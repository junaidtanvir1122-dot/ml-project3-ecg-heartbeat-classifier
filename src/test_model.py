import torch
import os
import matplotlib.pyplot as plt
from prepare import prepare
from model import ECGNet
from metrics import accuracy, confusion_matrix, per_class_accuracy

*_, Xte, yte = prepare()
model = ECGNet()
model.load_state_dict(torch.load("results/best_model.pt"))   # best epoch ke parameters
with torch.no_grad():
    pred = model(Xte).argmax(1)                              # sab se bare logit wali class

names = ["N", "S", "V", "F", "Q"]
C = confusion_matrix(yte, pred)
pca = per_class_accuracy(C)
print("test accuracy :", accuracy(yte, pred))
print("balanced acc  :", pca.mean().item())                  # paanchon per-class accuracies ka average
print("per-class acc :", {n: round(a, 4) for n, a in zip(names, pca.tolist())})
print("always-N acc  :", (yte == 0).float().mean().item())   # Task 6.1
print("confusion matrix (rows = true, cols = predicted):\n", C)
print("total:", C.sum().item(), "| row sums:", C.sum(1).tolist())

R = C / C.sum(1, keepdim=True)                               # har true class ka kitna hissa kahan gaya
plt.imshow(R, cmap="Blues")
for j in range(5):
    for k in range(5): plt.text(k, j, int(C[j, k]), ha="center", va="center", color="white" if R[j, k] > 0.5 else "black")
plt.xticks(range(5), names); plt.yticks(range(5), names)
plt.xlabel("predicted"); plt.ylabel("true"); plt.colorbar(label="share of true class")
os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/confusion_matrix.png", dpi=150)