import os, time, torch
import matplotlib.pyplot as plt
from prepare import prepare
from model import ECGNet, cross_entropy

B, BETA, LAM, EPOCHS = 128, 0.9, 1e-4, 30
dev = "cuda" if torch.cuda.is_available() else "cpu"
Xtr, ytr, Xv, yv, *_ = [t.to(dev) for t in prepare()]       # test data yahan chhoota bhi nahi

@torch.no_grad()
def evaluate(model, X, y):                                   # (loss, accuracy) poore X par
    z = model(X)
    return cross_entropy(z, y).item(), (z.argmax(1) == y).float().mean().item()

def train(alpha):
    model = ECGNet().to(dev)                                 # fresh network, seed 0
    params = list(model.W) + list(model.b)
    lams = [LAM] * len(model.W) + [0.0] * len(model.b)       # weight decay sirf weights par
    vel = [torch.zeros_like(p) for p in params]              # velocity zero se shuru
    h = {"tl": [], "vl": [], "ta": [], "va": [], "best_vl": float("inf")}
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr), device=dev)          # har epoch shuffle
        for i in range(0, len(Xtr), B):                      # ek mini-batch = ek update
            idx = perm[i:i + B]
            for p in params: p.grad = None                   # purane gradients saaf
            cross_entropy(model(Xtr[idx]), ytr[idx]).backward()
            with torch.no_grad():
                for p, v, lam in zip(params, vel, lams):
                    g = p.grad + lam * p                     # g = dJ/dP + lambda*P
                    v.mul_(BETA).add_((1 - BETA) * g)        # v = beta*v + (1-beta)*g
                    p.sub_(alpha * v)                        # P = P - alpha*v
        (tl, ta), (vl, va) = evaluate(model, Xtr, ytr), evaluate(model, Xv, yv)
        for k, val in zip(("tl", "vl", "ta", "va"), (tl, vl, ta, va)): h[k].append(val)
        if vl < h["best_vl"]:                                # ab tak ka sab se behtar epoch
            h["best_vl"], h["best_ep"] = vl, ep
            h["state"] = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        print(f"alpha={alpha} epoch {ep+1:2d}  train {tl:.4f}/{ta:.4f}  val {vl:.4f}/{va:.4f}")
    return h

if __name__ == "__main__":
    os.makedirs("results/figures", exist_ok=True)            # figures ka folder (na ho to bana do)
    print("initial loss (full train set):", evaluate(ECGNet().to(dev), Xtr, ytr)[0])
    runs = {}
    for a in (0.01, 0.1, 0.5):
        t0 = time.time(); runs[a] = train(a); runs[a]["time"] = time.time() - t0

    print("\nTable 3")
    for a, r in runs.items():
        print(f"alpha={a}: lowest val loss {r['best_vl']:.4f}, val acc {r['va'][r['best_ep']]:.4f}")
    best = min(runs, key=lambda a: runs[a]["best_vl"]); r = runs[best]; e = r["best_ep"]
    hw = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
    print(f"\nTable 2: alpha={best}, best epoch={e+1}, train loss/acc {r['tl'][e]:.4f}/{r['ta'][e]:.4f}, "
          f"val loss/acc {r['vl'][e]:.4f}/{r['va'][e]:.4f}, time {r['time']:.0f}s on {hw}")
    torch.save(r["state"], "results/best_model.pt")          # best epoch ke parameters

    ep = range(1, EPOCHS + 1)
    plt.figure()                                             # Figure 2: teeno alpha ki val loss
    for a, rr in runs.items(): plt.plot(ep, rr["vl"], label=f"alpha={a}")
    plt.xlabel("epoch"); plt.ylabel("validation loss"); plt.legend()
    plt.savefig("results/figures/lr_choice.png", dpi=150)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))            # Figure 3: chosen alpha ke curves
    for axis, k, name in ((ax[0], "l", "loss"), (ax[1], "a", "accuracy")):
        axis.plot(ep, r["t" + k], label="train"); axis.plot(ep, r["v" + k], label="validation")
        axis.set_xlabel("epoch"); axis.set_ylabel(name); axis.legend()
    fig.tight_layout(); fig.savefig("results/figures/curves.png", dpi=150)