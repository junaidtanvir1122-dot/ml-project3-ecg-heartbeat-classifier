import numpy as np, torch
from load_data import load_raw

def prepare(data_dir="data"):
    raw = load_raw(data_dir)
    train_df, test_df = raw["train"], raw["test"]
    X, y = train_df.iloc[:, :187].values, train_df[187].astype(int).values    # signal, label
    Xte, yte = test_df.iloc[:, :187].values, test_df[187].astype(int).values

    rng = np.random.default_rng(42)                      # assignment ka split, as it is
    idx = rng.permutation(len(train_df))
    n_val = int(0.1 * len(train_df))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    fx = lambda a: torch.tensor(a, dtype=torch.float32)  # signals -> float32
    fy = lambda a: torch.tensor(a, dtype=torch.long)     # labels  -> long
    return (fx(X[train_idx]), fy(y[train_idx]), fx(X[val_idx]), fy(y[val_idx]),
            fx(Xte), fy(yte))

if __name__ == "__main__":
    tr, te = load_raw()["train"], load_raw()["test"]
    sig = tr.iloc[:, :187].values
    print("shapes:", tr.shape, te.shape)
    print("label dtype:", tr[187].dtype)
    print("min / max:", sig.min(), sig.max())
    print("train per class:", np.bincount(tr[187].astype(int)))
    print("test per class: ", np.bincount(te[187].astype(int)))
    Xtr, ytr, Xv, yv, Xte, yte = prepare()
    print("train / val / test rows:", len(Xtr), len(Xv), len(Xte))