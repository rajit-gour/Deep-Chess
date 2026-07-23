# Trains the NN correction for the hybrid eval.
# The net predicts a cp offset; final eval = classical(pos) + net(pos).
# Loss compares sigmoid((classical+net)/400) to Stockfish's win prob, so the
# net only learns to fix what the classical eval gets wrong.
# out: net.txt (LAYERS format, linear output)
#   python train_hybrid.py --out net.txt --data data.csv data_big.csv data_sp.csv

import argparse
import numpy as np
import train   # encode_fen (768 side-to-move features), sigmoid

IN = 768
CP = 400.0
PIECE = "PNBRQK"

# same tables as engine_final.cpp
PVAL = [100, 320, 330, 500, 900, 0]
PST = np.array([
 [0,0,0,0,0,0,0,0, 50,50,50,50,50,50,50,50, 10,10,20,30,30,20,10,10,
  5,5,10,25,25,10,5,5, 0,0,0,20,20,0,0,0, 5,-5,-10,0,0,-10,-5,5,
  5,10,10,-20,-20,10,10,5, 0,0,0,0,0,0,0,0],
 [-50,-40,-30,-30,-30,-30,-40,-50, -40,-20,0,0,0,0,-20,-40, -30,0,10,15,15,10,0,-30,
  -30,5,15,20,20,15,5,-30, -30,0,15,20,20,15,0,-30, -30,5,10,15,15,10,5,-30,
  -40,-20,0,5,5,0,-20,-40, -50,-40,-30,-30,-30,-30,-40,-50],
 [-20,-10,-10,-10,-10,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,10,10,5,0,-10,
  -10,5,5,10,10,5,5,-10, -10,0,10,10,10,10,0,-10, -10,10,10,10,10,10,10,-10,
  -10,5,0,0,0,0,5,-10, -20,-10,-10,-10,-10,-10,-10,-20],
 [0,0,0,0,0,0,0,0, 5,10,10,10,10,10,10,5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5,
  -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, 0,0,0,5,5,0,0,0],
 [-20,-10,-10,-5,-5,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,5,5,5,0,-10,
  -5,0,5,5,5,5,0,-5, 0,0,5,5,5,5,0,-5, -10,5,5,5,5,5,0,-10,
  -10,0,5,0,0,0,0,-10, -20,-10,-10,-5,-5,-10,-10,-20],
 [-30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30, -20,-30,-30,-40,-40,-30,-30,-20, -10,-20,-20,-20,-20,-20,-20,-10,
  20,20,0,0,0,0,20,20, 20,30,10,0,0,10,30,20]], dtype=np.float32)


def classical_cp(fen):
    board, side = fen.split()[0], fen.split()[1]
    sc = 0; rank, file = 7, 0
    for ch in board:
        if ch == "/": rank -= 1; file = 0
        elif ch.isdigit(): file += int(ch)
        else:
            sq = rank * 8 + file; i = PIECE.index(ch.upper()); white = ch.isupper()
            v = PVAL[i] + PST[i][(sq ^ 56) if white else sq]
            sc += v if white else -v
            file += 1
    return sc if side == "w" else -sc


def load(paths, clamp=2000.0):
    X, base, y = [], [], []
    for p in paths:
        with open(p) as f:
            next(f)
            for line in f:
                line = line.rstrip("\n")
                if not line: continue
                fen, cp = line.rsplit(",", 1)
                X.append(train.encode_fen(fen))
                base.append(classical_cp(fen))
                y.append(max(-clamp, min(clamp, float(cp))))
    return (np.array(X, np.float32), np.array(base, np.float32).reshape(-1, 1),
            np.array(y, np.float32).reshape(-1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="net.txt")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=150)
    a = ap.parse_args()

    print("encoding...")
    X, base, ycp = load(a.data)
    n = X.shape[0]; H = a.hidden
    # regress the residual directly (in pawns), strong gradients so it learns
    resid = np.clip(ycp - base, -400, 400) / 100.0     # target in pawns
    print(f"{n} positions")

    rng = np.random.default_rng(0)
    W1 = (rng.standard_normal((IN, H)) * np.sqrt(2/IN)).astype(np.float32); b1 = np.zeros(H, np.float32)
    W2 = (rng.standard_normal((H, 1)) * np.sqrt(1/H)).astype(np.float32); b2 = np.zeros(1, np.float32)
    vW1=np.zeros_like(W1); vb1=np.zeros_like(b1); vW2=np.zeros_like(W2); vb2=np.zeros_like(b2)
    lr, mu, bs = 0.05, 0.9, 8192
    cut = int(n*0.97); idx = rng.permutation(n); tr, va = idx[:cut], idx[cut:]

    def vloss():
        h = np.maximum(X[va]@W1+b1, 0); out = h@W2+b2
        return float(np.mean((out-resid[va])**2))

    for ep in range(1, a.epochs+1):
        if ep % 50 == 0: lr *= 0.5
        order = rng.permutation(len(tr))
        for s in range(0, len(order), bs):
            b = tr[order[s:s+bs]]
            xb, tb = X[b], resid[b]
            h = np.maximum(xb@W1+b1, 0); out = h@W2+b2
            B = xb.shape[0]
            dout = (2.0/B)*(out-tb)                  # MSE on residual (pawns)
            dW2 = h.T@dout; db2 = dout.sum(0)
            dh = dout@W2.T; dz1 = dh*(h>0)
            dW1 = xb.T@dz1; db1 = dz1.sum(0)
            vW1=mu*vW1-lr*dW1; vb1=mu*vb1-lr*db1; vW2=mu*vW2-lr*dW2; vb2=mu*vb2-lr*db2
            W1+=vW1; b1+=vb1; W2+=vW2; b2+=vb2
        if ep % 10 == 0 or ep == 1: print(f"epoch {ep:3d} val {vloss():.5f}", flush=True)

    # export with the output layer scaled x100 so the engine reads centipawns
    with open(a.out, "w") as f:
        f.write("LAYERS 2\n")
        f.write(f"{IN} {H}\n")
        f.write(" ".join(f"{v:.6f}" for v in W1.reshape(-1)) + "\n")
        f.write(" ".join(f"{v:.6f}" for v in b1) + "\n")
        f.write(f"{H} 1\n")
        f.write(" ".join(f"{v:.6f}" for v in (W2*100).reshape(-1)) + "\n")
        f.write(" ".join(f"{v:.6f}" for v in b2*100) + "\n")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
