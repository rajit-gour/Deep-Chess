"""
Generate REALISTIC training positions via self-play of the strong classical
engine (engine_v3 with no weights). Random opening plies + occasional random
moves give diversity; the rest are real engine moves, so the positions look
like real games instead of the random-move garbage we trained on before.

Fixed search DEPTH per move (not time) so this is insensitive to CPU load.

OUT: fens_sp.txt (one FEN per line)
USAGE: python selfplay_gen.py 40000 fens_sp.txt 6 3
         arg1 count, arg2 out, arg3 engine depth, arg4 random opening plies
"""
import sys, random
import chess, chess.engine

ENGINE = ["./engine_v3.exe", "none"]   # classical eval


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    out = sys.argv[2] if len(sys.argv) > 2 else "fens_sp.txt"
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    open_plies = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    rng = random.Random(1234)

    eng = chess.engine.SimpleEngine.popen_uci(ENGINE)
    written = 0
    f = open(out, "w")
    try:
        while written < want:
            board = chess.Board()
            # random opening for diversity
            for _ in range(open_plies):
                mv = list(board.legal_moves)
                if not mv: break
                board.push(rng.choice(mv))
            # play a real game, sampling positions
            for ply in range(200):
                if board.is_game_over(claim_draw=True) or written >= want:
                    break
                # 10% random move keeps distribution broad
                if rng.random() < 0.10:
                    board.push(rng.choice(list(board.legal_moves)))
                else:
                    r = eng.play(board, chess.engine.Limit(depth=depth))
                    if r.move is None: break
                    board.push(r.move)
                # sample quiet-ish positions (not in check)
                if ply >= 1 and not board.is_check():
                    f.write(board.fen() + "\n")
                    written += 1
                    if written % 2000 == 0:
                        print(f"  {written}/{want}", end="\r", flush=True)
    finally:
        eng.quit(); f.close()
    print(f"\nWrote {written} self-play positions to {out}")


if __name__ == "__main__":
    main()
