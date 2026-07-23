"""
Deeper trainer: a multi-layer MLP, still PLAIN NUMPY (no PyTorch, no autograd).
We generalise the single-hidden-layer net from train.py to an arbitrary stack
of fully-connected layers, and implement forward + backprop for the whole stack
with two simple loops.

    ARCHITECTURE (SIZES below):   768 -> 512 -> 32 -> 1
      layer 0:  768 -> 512   ReLU
      layer 1:  512 ->  32   ReLU
      layer 2:   32 ->   1   (raw), then sigmoid -> win probability

Everything else (feature encoding, cp->winprob target, loss) is identical to
train.py, which we import so there's a single source of truth for the encoding.

WEIGHT FILE FORMAT (generic, read by engine_nn.cpp):
    LAYERS <N>
    <in> <out>              # layer 0
    <in*out floats>         #   W row-major:  W[i*out + j]
    <out floats>            #   b
    <in> <out>              # layer 1
    ...
RUN:
    python train_deep.py data.csv weights.txt
"""

import sys
import numpy as np
import train   # reuse encode_fen, cp_to_winprob, load_data, sigmoid

SIZES = [768, 512, 32, 1]     # <-- change this list to reshape the net


def init_params(sizes, seed=0):
    rng = np.random.default_rng(seed)
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        # He init for ReLU layers; the last layer is linear but He is fine too
        W = (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
        bias = np.zeros((b,), dtype=np.float32)
        params.append([W, bias])
    return params


def forward(params, x):
    """Return (p, cache). p = win prob. cache holds per-layer (input, preact)
    so backprop can reuse them."""
    a = x
    cache = []
    L = len(params)
    for l, (W, b) in enumerate(params):
        z = a @ W + b
        cache.append((a, z))
        if l < L - 1:
            a = np.maximum(z, 0.0)        # ReLU on hidden layers
        else:
            a = train.sigmoid(z)          # sigmoid on the final layer
    return a, cache


def backward(params, cache, p, t):
    """Return gradients [(dW,db), ...] for MSE loss ( (p-t)^2 ) averaged over
    the batch. Standard backprop: start from the output, walk layers backwards."""
    B = p.shape[0]
    L = len(params)
    grads = [None] * L

    # dL/dz at the output layer: MSE through the sigmoid
    #   L = mean((p-t)^2);  dL/dp = 2(p-t)/B;  dp/dz = p(1-p)
    dz = (2.0 / B) * (p - t) * p * (1.0 - p)

    for l in range(L - 1, -1, -1):
        a_in, z = cache[l]
        W, b = params[l]
        dW = a_in.T @ dz
        db = dz.sum(axis=0)
        grads[l] = (dW, db)
        if l > 0:
            da_prev = dz @ W.T
            # backprop through the previous layer's ReLU
            _, z_prev = cache[l - 1]
            dz = da_prev * (z_prev > 0.0)
    return grads


def train_net(X, y, sizes=SIZES, epochs=60, batch=4096, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    params = init_params(sizes, seed)

    n = X.shape[0]
    idx = rng.permutation(n)
    cut = int(n * 0.95)
    tr, va = idx[:cut], idx[cut:]
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]

    def mse(Xs, ys):
        p, _ = forward(params, Xs)
        return float(np.mean((p - ys) ** 2))

    for ep in range(1, epochs + 1):
        order = rng.permutation(Xtr.shape[0])
        for s in range(0, len(order), batch):
            bi = order[s:s + batch]
            xb, tb = Xtr[bi], ytr[bi]
            p, cache = forward(params, xb)
            grads = backward(params, cache, p, tb)
            for (W, b), (dW, db) in zip(params, grads):
                W -= lr * dW
                b -= lr * db
        print(f"epoch {ep:3d}  train {mse(Xtr, ytr):.5f}  val {mse(Xva, yva):.5f}")

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
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "weights.txt"
    X, y = train.load_data(csv_path)     # same encoding as the single-layer net
    params = train_net(X, y)
    save_weights(out_path, params)


if __name__ == "__main__":
    main()
