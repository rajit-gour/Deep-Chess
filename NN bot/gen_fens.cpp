// STEP 1 of the NN pipeline: generate a diverse set of board positions.
//
// We produce positions by playing RANDOM legal moves (random self-play) from
// the start position. Random games wander into all sorts of middlegames and
// endgames, which gives the network broad coverage of what a board can look
// like. We don't care that the moves are bad -- we only want varied positions;
// Stockfish (step 2) will tell us how good each position actually is.
//
// Output: one FEN per line to a file (default fens.txt).
//
// BUILD:  g++ -std=c++17 -O3 -static gen_fens.cpp -o gen_fens.exe
// RUN:    ./gen_fens.exe 150000 fens.txt 12345
//           arg1 = how many positions to write   (default 20000)
//           arg2 = output file                    (default fens.txt)
//           arg3 = random seed                    (default 42)

#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include "chess.hpp"

using namespace chess;
using namespace std;

int main(int argc, char** argv) {
    long long want = (argc > 1) ? stoll(argv[1]) : 20000;
    string outpath = (argc > 2) ? argv[2] : "fens.txt";
    unsigned seed = (argc > 3) ? (unsigned)stoul(argv[3]) : 42u;

    mt19937 rng(seed);
    ofstream out(outpath);
    if (!out) { cerr << "cannot open " << outpath << "\n"; return 1; }

    long long written = 0;
    Board board;

    while (written < want) {
        // start a fresh random game
        board.setFen(constants::STARTPOS);

        // play a random game up to ~120 plies, sampling positions along the way
        for (int ply = 0; ply < 120 && written < want; ply++) {
            // stop this game if it's over (checkmate / stalemate / draw)
            auto [reason, result] = board.isGameOver();
            if (result != GameResult::NONE) break;

            Movelist moves;
            movegen::legalmoves(moves, board);
            if (moves.empty()) break;

            // Sample a position only after a few opening moves (skip the very
            // first plies -- they're all identical across games) and not on
            // every single ply (consecutive plies are almost the same board).
            // We record ~1 in 2 positions once past ply 6.
            if (ply >= 6 && (rng() & 1)) {
                out << board.getFen() << "\n";
                written++;
                if (written % 5000 == 0)
                    cerr << "written " << written << " / " << want << "\r";
            }

            // pick a uniformly random legal move and play it
            uniform_int_distribution<int> pick(0, moves.size() - 1);
            board.makeMove(moves[pick(rng)]);
        }
    }

    out.close();
    cerr << "\nDone. Wrote " << written << " positions to " << outpath << "\n";
    return 0;
}
