"""
Stronger NumPy trainer (still no PyTorch):
  * momentum SGD          (smoother, faster convergence than plain SGD)
  * step learning-rate decay
  * more epochs
  * combine multiple data CSVs into one training set
  * configurable architecture via --sizes

Reuses train.encode_fen so the feature encoding stays identical to the engine.

EXAMPLES
  # deep net on the combined 200k dataset
  python train_final.py --out weights.txt --data data.csv data_big.csv \
         --sizes 768 512 32 1 --epochs 150

  # single-hidden-layer baseline on the same data
  python train_final.py --out weights_base.txt --data data.csv data_big.csv \
         --sizes 768 256 1 --epochs 150
"""

import argparse
import numpy as np
import train   # encode_fen, cp_to_winprob, sigmoid

IN = 768


def load_many(paths, cp_clamp=2000.0):
    fens, cps = [], []
    for p in paths:
        with open(p) as f:
            next(f)
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                fen, cp = line.rsplit(",", 1)
                fens.append(fen)
                cps.append(float(cp))
    n = len(fens)
    print(f"Loaded {n} positions from {len(paths)} file(s). Encoding ...")
    X = np.zeros((n, IN), dtype=np.float32)
    for i, fen in enumerate(fens):
        X[i] = train.encode_fen(fen)
    cp = np.clip(np.array(cps, dtype=np.float32), -cp_clamp, cp_clamp)
    y = train.cp_to_winprob(cp).reshape(-1, 1)
    return X, y


def init_params(sizes, seed=0):
    rng = np.random.default_rng(seed)
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        W = (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
        bias = np.zeros((b,), dtype=np.float32)
        params.append([W, bias])
    return params


def forward(params, x):
    a = x
    cache = []
    L = len(params)
    for l, (W, b) in enumerate(params):
        z = a @ W + b
        cache.append((a, z))
        a = np.maximum(z, 0.0) if l < L - 1 else train.sigmoid(z)
    return a, cache


def backward(params, cache, p, t):
    B = p.shape[0]
    L = len(params)
    grads = [None] * L
    dz = (2.0 / B) * (p - t) * p * (1.0 - p)
    for l in range(L - 1, -1, -1):
        a_in, z = cache[l]
        W, _ = params[l]
        grads[l] = (a_in.T @ dz, dz.sum(axis=0))
        if l > 0:
            _, z_prev = cache[l - 1]
            dz = (dz @ W.T) * (z_prev > 0.0)
    return grads


def train_net(X, y, sizes, epochs, batch=4096, lr=0.1, mu=0.9,
              decay_every=50, decay=0.5, seed=0):
    rng = np.random.default_rng(seed)
    params = init_params(sizes, seed)
    vel = [[np.zeros_like(W), np.zeros_like(b)] for W, b in params]

    n = X.shape[0]
    idx = rng.permutation(n)
    cut = int(n * 0.95)
    tr, va = idx[:cut], idx[cut:]
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]

    def mse(Xs, ys):
        p, _ = forward(params, Xs)
        return float(np.mean((p - ys) ** 2))

    best_val = 1e9
    for ep in range(1, epochs + 1):
        if ep % decay_every == 0:
            lr *= decay
        order = rng.permutation(Xtr.shape[0])
        for s in range(0, len(order), batch):
            bi = order[s:s + batch]
            p, cache = forward(params, Xtr[bi])
            grads = backward(params, cache, p, ytr[bi])
            for (W, b), (vW, vb), (dW, db) in zip(params, vel, grads):
                vW[...] = mu * vW - lr * dW
                vb[...] = mu * vb - lr * db
                W += vW
                b += vb
        v = mse(Xva, yva)
        best_val = min(best_val, v)
        if ep % 10 == 0 or ep == 1:
            print(f"epoch {ep:3d}  lr {lr:.4f}  train {mse(Xtr, ytr):.5f}  val {v:.5f}")
    print(f"best val {best_val:.5f}")
    return params


def save_weights(path, params):
    with open(path, "w") as f:
        f.write(f"LAYERS {len(params)}\n")
        for W, b in params:
            a, o = W.shape
            f.write(f"{a} {o}\n")
            f.write(" ".join(f"{v:.6f}" for v in W.reshape(-1)) + "\n")
            f.write(" ".join(f"{v:.6f}" for v in b.reshape(-1)) + "\n")
    print(f"Saved {len(params)}-layer net to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="weights.txt")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--sizes", nargs="+", type=int, default=[768, 512, 32, 1])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.1)
    args = ap.parse_args()

    X, y = load_many(args.data)
    params = train_net(X, y, args.sizes, args.epochs, lr=args.lr)
    save_weights(args.out, params)


if __name__ == "__main__":
    main()
