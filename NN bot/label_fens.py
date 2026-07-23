"""
STEP 2 of the NN pipeline: label each FEN with Stockfish's evaluation.

We open ONE long-lived Stockfish process and talk to it over its stdin/stdout
using the UCI protocol (the same protocol your own engine speaks). For each
position we send:
    position fen <FEN>
    go depth <D>
and read back lines like:
    info ... score cp 34 ...      (34 centipawns, side-to-move's view)
    info ... score mate 3 ...     (mate in 3 for side to move)
    bestmove e2e4
The last "score" seen before "bestmove" is Stockfish's verdict.

IMPORTANT sign convention: UCI "score cp" is ALWAYS from the perspective of the
side to move (positive = good for whoever is about to move). Your C++
evaluate() also returns a score relative to the side to move, so the two line
up perfectly -- no flipping needed here.

Output: data.csv with columns  fen,cp   (cp = centipawns, side-to-move view)

RUN:
    python label_fens.py fens.txt data.csv 10
        arg1 = input FENs      (default fens.txt)
        arg2 = output csv       (default data.csv)
        arg3 = search depth     (default 10)  -- lower = faster, noisier labels
"""

import subprocess
import sys
import csv

STOCKFISH = r"C:\Users\Rajit\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"

# centipawn value we assign to a forced mate (bigger than any normal eval so
# the network learns "mate = decisively winning"). Scaled slightly by distance
# so mate-in-1 looks better than mate-in-10.
MATE_CP = 10000


def parse_score(line: str):
    """Return centipawns (side-to-move view) from an 'info ... score ...' line,
    or None if this line has no score."""
    toks = line.split()
    if "score" not in toks:
        return None
    i = toks.index("score")
    kind = toks[i + 1]
    val = int(toks[i + 2])
    if kind == "cp":
        return val
    if kind == "mate":
        # val>0: we mate them; val<0: they mate us. Subtract distance so closer
        # mates score higher in magnitude.
        return (MATE_CP - abs(val)) * (1 if val > 0 else -1)
    return None


def main():
    fens_path = sys.argv[1] if len(sys.argv) > 1 else "fens.txt"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "data.csv"
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    sf = subprocess.Popen(
        [STOCKFISH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def send(cmd):
        sf.stdin.write(cmd + "\n")
        sf.stdin.flush()

    # handshake + a little muscle: more threads/hash = faster labelling
    send("uci")
    while True:
        if sf.stdout.readline().strip() == "uciok":
            break
    send("setoption name Threads value 4")
    send("setoption name Hash value 256")
    send("isready")
    while True:
        if sf.stdout.readline().strip() == "readyok":
            break

    with open(fens_path) as f:
        fens = [ln.strip() for ln in f if ln.strip()]

    total = len(fens)
    print(f"Labelling {total} positions at depth {depth} ...")

    with open(out_path, "w", newline="") as csvf:
        w = csv.writer(csvf)
        w.writerow(["fen", "cp"])

        for n, fen in enumerate(fens, 1):
            send(f"position fen {fen}")
            send(f"go depth {depth}")

            last_cp = None
            while True:
                line = sf.stdout.readline()
                if not line:
                    break
                s = parse_score(line)
                if s is not None:
                    last_cp = s
                if line.startswith("bestmove"):
                    break

            if last_cp is not None:
                w.writerow([fen, last_cp])

            if n % 1000 == 0:
                print(f"  {n}/{total}", end="\r", flush=True)

    send("quit")
    sf.wait()
    print(f"\nDone. Wrote labels to {out_path}")


if __name__ == "__main__":
    main()
