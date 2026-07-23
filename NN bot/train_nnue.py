"""
Train a real (small) NNUE in PyTorch.

ARCHITECTURE
  feature transformer  FT : 768 -> 256   (shared weights, applied to BOTH sides)
  accumulator(perspective) = FT_bias + sum of FT rows of that side's active features
  eval input = [ clippedReLU(own_acc) , clippedReLU(opp_acc) ]   (512)
             -> L1: 512 -> 32 -> clippedReLU
             -> L2:  32 ->  1 -> sigmoid  => win probability
  target = sigmoid(cp/400)

TWO perspective-INDEPENDENT accumulators (own+opp) are what make the C++ eval
incrementally updatable (add/subtract the one piece that moved). See engine_nnue.cpp.

FEATURE INDEX (perspective P) -- must match engine_nnue.cpp EXACTLY:
  piece color C, type T(0..5=PNBRQK), square S(a1=0..h8=63)
     own=(C==P); plane = own?T:T+6; sqp=(P==White)?S:S^56; index = plane*64+sqp

We pad each position's active-feature list to 32 slots with PAD=768 (a zero row),
so a whole batch is one dense (B,32) tensor -> fast vectorised EmbeddingBag.

RUN: python train_nnue.py --out nnue.txt --data data.csv data_big.csv data_sp.csv --epochs 120
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

IN, PAD = 768, 768       # PAD is an extra zero row at index 768
ACC = 256                # accumulator size (overridden by --acc)
CP_SCALE = 400.0
PIECE = "PNBRQK"
MAXF = 32


def encode_indices(fen):
    board, side = fen.split()[0], fen.split()[1]
    w = [PAD] * MAXF
    b = [PAD] * MAXF
    k = 0
    rank, file = 7, 0
    for ch in board:
        if ch == "/":
            rank -= 1; file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            sq = rank * 8 + file
            T = PIECE.index(ch.upper())
            is_white = ch.isupper()
            w[k] = (T if is_white else T + 6) * 64 + sq
            b[k] = (T if (not is_white) else T + 6) * 64 + (sq ^ 56)
            k += 1
            file += 1
    return w, b, 1.0 if side == "w" else 0.0


def load(paths, cp_clamp=2000.0):
    W, B, S, Y = [], [], [], []
    for p in paths:
        with open(p) as f:
            next(f)
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                fen, cp = line.rsplit(",", 1)
                w, b, stm = encode_indices(fen)
                W.append(w); B.append(b); S.append(stm)
                cpv = max(-cp_clamp, min(cp_clamp, float(cp)))
                Y.append(1.0 / (1.0 + np.exp(-cpv / CP_SCALE)))
    return (torch.tensor(W, dtype=torch.long), torch.tensor(B, dtype=torch.long),
            torch.tensor(S, dtype=torch.float32).unsqueeze(1),
            torch.tensor(Y, dtype=torch.float32).unsqueeze(1))


class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = nn.EmbeddingBag(IN + 1, ACC, mode="sum", padding_idx=PAD)
        self.ft_bias = nn.Parameter(torch.zeros(ACC))
        self.l1 = nn.Linear(2 * ACC, 32)
        self.l2 = nn.Linear(32, 1)

    def forward(self, w, b, stm):
        aw = self.ft(w) + self.ft_bias     # (B,ACC)  White-perspective accumulator
        ab = self.ft(b) + self.ft_bias     # (B,ACC)  Black-perspective accumulator
        own = stm * aw + (1 - stm) * ab
        opp = stm * ab + (1 - stm) * aw
        # plain ReLU for float training (clamped-ReLU is for the quantized build)
        h = torch.cat([own.clamp(min=0), opp.clamp(min=0)], dim=1)
        h = self.l1(h).clamp(min=0)
        return torch.sigmoid(self.l2(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="nnue.txt")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--acc", type=int, default=256)
    a = ap.parse_args()
    global ACC
    ACC = a.acc

    print("loading + encoding ...")
    W, B, S, Y = load(a.data)
    n = S.shape[0]; print(f"{n} positions")
    torch.set_num_threads(torch.get_num_threads())

    model = NNUE()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.5)
    lossf = nn.MSELoss()
    gen = np.random.default_rng(0)
    cut = int(n * 0.97); perm = gen.permutation(n); tr, va = perm[:cut], perm[cut:]
    trW, trB, trS, trY = W[tr], B[tr], S[tr], Y[tr]
    vaW, vaB, vaS, vaY = W[va], B[va], S[va], Y[va]

    best = 1e9
    for ep in range(1, a.epochs + 1):
        model.train()
        order = torch.randperm(trS.shape[0])
        for s in range(0, order.shape[0], a.bs):
            i = order[s:s + a.bs]
            pred = model(trW[i], trB[i], trS[i])
            loss = lossf(pred, trY[i])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 10 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                vl = lossf(model(vaW, vaB, vaS), vaY).item()
            best = min(best, vl)
            print(f"epoch {ep:3d}  val {vl:.5f}", flush=True)
    print(f"best val {best:.5f}")

    with torch.no_grad():
        ftW = model.ft.weight.cpu().numpy()[:IN]   # drop PAD row -> (768,256)
        ftB = model.ft_bias.cpu().numpy()
        l1W = model.l1.weight.cpu().numpy(); l1B = model.l1.bias.cpu().numpy()
        l2W = model.l2.weight.cpu().numpy(); l2B = model.l2.bias.cpu().numpy()
    with open(a.out, "w") as f:
        f.write(f"NNUE {IN} {ACC}\n")
        f.write(" ".join(f"{v:.6f}" for v in ftW.reshape(-1)) + "\n")
        f.write(" ".join(f"{v:.6f}" for v in ftB) + "\n")
        f.write(" ".join(f"{v:.6f}" for v in l1W.reshape(-1)) + "\n")
        f.write(" ".join(f"{v:.6f}" for v in l1B) + "\n")
        f.write(" ".join(f"{v:.6f}" for v in l2W.reshape(-1)) + "\n")
        f.write(" ".join(f"{v:.6f}" for v in l2B) + "\n")
    print(f"saved NNUE to {a.out}")


if __name__ == "__main__":
    main()
