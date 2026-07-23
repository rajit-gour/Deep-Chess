"""
STEP 3 of the NN pipeline: train a neural network to predict board utility.

This is written in PLAIN NUMPY -- no PyTorch, no autograd. We implement the
forward pass, the loss, and backpropagation (the gradients) BY HAND, then do
gradient descent. The point this week is to understand exactly what a NN does.

------------------------------------------------------------------------------
THE NETWORK          768  ->  256 (ReLU)  ->  1
------------------------------------------------------------------------------
  input  x : 768 numbers, each 0 or 1  (the board, see encode_fen below)
  hidden h : 256 numbers               h = relu(W1 @ x + b1)
  output y : 1 number (raw score)      y = W2 @ h + b2

We squash y through a sigmoid to a win-probability p in (0,1) and compare it to
Stockfish's win-probability target t. Loss = mean( (p - t)^2 )  (MSE).

Why win-probability instead of raw centipawns? Raw cp ranges from -10000 to
+10000; the difference between +900 and +1000 cp barely matters (both winning)
but MSE would treat it as huge. Mapping cp -> win% with a sigmoid compresses
the extremes and focuses learning on positions that are actually close.
    t = sigmoid(cp / 400) = 1 / (1 + 10^(-cp/400))        [we use base-e below]

------------------------------------------------------------------------------
FEATURE ENCODING  (MUST match engine_nn.cpp EXACTLY, or the net is useless)
------------------------------------------------------------------------------
12 "planes" of 64 squares = 768 inputs. Planes are ordered:
    0..5  : OWN   pawn,knight,bishop,rook,queen,king   (own = side to move)
    6..11 : ENEMY pawn,knight,bishop,rook,queen,king
Square numbering matches chess.hpp: a1=0, b1=1, ..., h8=63 (sq = rank*8 + file).

PERSPECTIVE FLIP: we always encode from the side-to-move's point of view, so the
net learns ONE function ("how good is it for me") instead of two. If it's black
to move we (a) treat black pieces as "own", and (b) mirror every square
vertically with sq ^ 56 so black's home rank looks like white's home rank.
This matches the side-to-move sign convention of the cp labels and of your
C++ evaluate().

    index = plane * 64 + (sq if white_to_move else sq ^ 56)

RUN:
    python train.py data.csv weights.txt
"""

import sys
import numpy as np

# ---- fixed sizes -----------------------------------------------------------
IN, H, OUT = 768, 256, 1
CP_SCALE = 400.0          # cp -> win% steepness (same constant used in C++)

PIECE_ORDER = "PNBRQK"    # index 0..5 within a color


# ---------------------------------------------------------------------------
# FEATURE ENCODING
# ---------------------------------------------------------------------------
def encode_fen(fen: str) -> np.ndarray:
    """Turn a FEN string into a 768-length float32 vector of 0s and 1s.
    Mirrors engine_nn.cpp::encode() exactly."""
    x = np.zeros(IN, dtype=np.float32)
    board_part, side = fen.split()[0], fen.split()[1]
    white_to_move = (side == "w")

    # FEN lists rank 8 first, down to rank 1. Walk it and compute a1=0 squares.
    rank = 7
    file = 0
    for ch in board_part:
        if ch == "/":
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)          # empty squares
        else:
            sq = rank * 8 + file     # chess.hpp numbering: a1=0 ... h8=63
            is_white_piece = ch.isupper()
            piece_idx = PIECE_ORDER.index(ch.upper())   # 0..5

            # "own" = side to move
            own = (is_white_piece == white_to_move)
            plane = piece_idx if own else piece_idx + 6

            sq_persp = sq if white_to_move else (sq ^ 56)  # vertical mirror
            x[plane * 64 + sq_persp] = 1.0
            file += 1
    return x


def cp_to_winprob(cp: np.ndarray) -> np.ndarray:
    """Stockfish centipawns -> win probability in (0,1), side-to-move view."""
    return 1.0 / (1.0 + np.exp(-cp / CP_SCALE))


# ---------------------------------------------------------------------------
# LOAD + ENCODE DATA
# ---------------------------------------------------------------------------
def load_data(csv_path):
    fens, cps = [], []
    with open(csv_path) as f:
        next(f)  # header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # fen may contain commas? No -- FEN has no commas, safe to rsplit.
            fen, cp = line.rsplit(",", 1)
            fens.append(fen)
            cps.append(float(cp))

    n = len(fens)
    print(f"Encoding {n} positions ...")
    X = np.zeros((n, IN), dtype=np.float32)
    for i, fen in enumerate(fens):
        X[i] = encode_fen(fen)
    y = cp_to_winprob(np.array(cps, dtype=np.float32)).reshape(-1, 1)
    return X, y


# ---------------------------------------------------------------------------
# THE NETWORK (forward + backward, by hand)
# ---------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train(X, y, epochs=60, batch=4096, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    # He initialization for the ReLU layer; small for the output layer.
    W1 = (rng.standard_normal((IN, H)) * np.sqrt(2.0 / IN)).astype(np.float32)
    b1 = np.zeros((H,), dtype=np.float32)
    W2 = (rng.standard_normal((H, OUT)) * np.sqrt(1.0 / H)).astype(np.float32)
    b2 = np.zeros((OUT,), dtype=np.float32)

    # simple 95/5 train/validation split so we can watch for overfitting
    idx = rng.permutation(n)
    cut = int(n * 0.95)
    tr, va = idx[:cut], idx[cut:]
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]

    for ep in range(1, epochs + 1):
        order = rng.permutation(Xtr.shape[0])
        for s in range(0, len(order), batch):
            bi = order[s:s + batch]
            xb = Xtr[bi]                    # (B, 768)
            tb = ytr[bi]                    # (B, 1)
            B = xb.shape[0]

            # ---- forward ----
            z1 = xb @ W1 + b1               # (B, 256)
            h = np.maximum(z1, 0.0)         # ReLU
            z2 = h @ W2 + b2                # (B, 1)  raw score
            p = sigmoid(z2)                 # (B, 1)  win prob

            # ---- loss = mean( (p - t)^2 ) ; backprop the gradients ----
            # dL/dp = 2(p - t)/B ; dp/dz2 = p(1-p)  -> chain them:
            dz2 = (2.0 / B) * (p - tb) * p * (1.0 - p)   # (B, 1)
            dW2 = h.T @ dz2                               # (256, 1)
            db2 = dz2.sum(axis=0)                         # (1,)

            dh = dz2 @ W2.T                               # (B, 256)
            dz1 = dh * (z1 > 0.0)                         # ReLU derivative
            dW1 = xb.T @ dz1                              # (768, 256)
            db1 = dz1.sum(axis=0)                         # (256,)

            # ---- gradient descent step ----
            W1 -= lr * dW1; b1 -= lr * db1
            W2 -= lr * dW2; b2 -= lr * db2

        # ---- report train/val loss each epoch ----
        def mse(Xs, ys):
            hh = np.maximum(Xs @ W1 + b1, 0.0)
            pp = sigmoid(hh @ W2 + b2)
            return float(np.mean((pp - ys) ** 2))
        print(f"epoch {ep:3d}  train {mse(Xtr, ytr):.5f}  val {mse(Xva, yva):.5f}")

    return W1, b1, W2, b2


# ---------------------------------------------------------------------------
# EXPORT WEIGHTS for the C++ engine
# ---------------------------------------------------------------------------
def save_weights(path, W1, b1, W2, b2):
    """Write a plain-text file the C++ engine reads with operator>>.
    Order: dims, then W1 (row-major IN*H), b1 (H), W2 (H*OUT), b2 (OUT)."""
    with open(path, "w") as f:
        f.write(f"{IN} {H} {OUT}\n")
        f.write(" ".join(f"{v:.6f}" for v in W1.reshape(-1)) + "\n")
        f.write(" ".join(f"{v:.6f}" for v in b1.reshape(-1)) + "\n")
        f.write(" ".join(f"{v:.6f}" for v in W2.reshape(-1)) + "\n")
        f.write(" ".join(f"{v:.6f}" for v in b2.reshape(-1)) + "\n")
    print(f"Saved weights to {path}")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "weights.txt"
    X, y = load_data(csv_path)
    W1, b1, W2, b2 = train(X, y)
    save_weights(out_path, W1, b1, W2, b2)


if __name__ == "__main__":
    main()
