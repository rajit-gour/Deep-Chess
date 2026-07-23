// engine_v2: NN evaluation + serious search.
//
// Over engine_nn.cpp this adds, all classical search improvements (the NN eval
// is unchanged):
//   * QUIESCENCE SEARCH  -- at the leaves, keep searching captures (and check
//     evasions) until the position is "quiet", so we never evaluate in the
//     middle of a trade. This removes the horizon-effect blunders.
//   * TRANSPOSITION TABLE -- cache each searched position by its Zobrist hash
//     (board.hash()) with depth/bound/best-move, so repeated positions are
//     instant and we get a great first move to try.
//   * MOVE ORDERING -- TT move, then MVV-LVA captures, then two "killer" quiet
//     moves per ply, then a history-heuristic score. Better ordering => more
//     alpha-beta cutoffs => deeper search in the same time.
//   * CHECK EXTENSION + draw detection (repetition / 50-move).
//
// BUILD:  g++ -std=c++17 -O3 -static engine_v2.cpp -o engine_v2.exe
// RUN:    ./engine_v2.exe [weights.txt]

#include <fstream>
#include <iostream>
#include <sstream>
#include "chess.hpp"
#include <cmath>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cstdint>

using namespace chess;
using namespace std;

// ===========================================================================
//  NEURAL NETWORK  (identical to engine_nn.cpp)
// ===========================================================================
struct NNUE {
    static const int IN = 768;
    static constexpr float CP_SCALE = 400.0f;
    struct Layer { int in, out; vector<float> W, b; };
    vector<Layer> layers;
    bool loaded = false;

    bool load(const string& path) {
        ifstream f(path);
        if (!f) return false;
        string tag; f >> tag;
        if (tag == "LAYERS") {
            int n; f >> n; layers.resize(n);
            for (auto& L : layers) {
                f >> L.in >> L.out;
                L.W.resize((size_t)L.in * L.out); L.b.resize(L.out);
                for (auto& v : L.W) f >> v;
                for (auto& v : L.b) f >> v;
            }
        } else {
            int in = stoi(tag), h, out; f >> h >> out;
            layers.resize(2);
            layers[0] = {in, h, vector<float>((size_t)in * h), vector<float>(h)};
            layers[1] = {h, out, vector<float>((size_t)h * out), vector<float>(out)};
            for (auto& v : layers[0].W) f >> v;
            for (auto& v : layers[0].b) f >> v;
            for (auto& v : layers[1].W) f >> v;
            for (auto& v : layers[1].b) f >> v;
        }
        loaded = (bool)f && !layers.empty() && layers.front().in == IN;
        return loaded;
    }

    static int pieceIndex(PieceType t) {
        if (t == PieceType::PAWN)   return 0;
        if (t == PieceType::KNIGHT) return 1;
        if (t == PieceType::BISHOP) return 2;
        if (t == PieceType::ROOK)   return 3;
        if (t == PieceType::QUEEN)  return 4;
        return 5;
    }

    // Forward pass with NO heap allocation (stack scratch buffers), so it is
    // fast enough to call at millions of search nodes. Max layer width <= 1024.
    static const int MAXW = 1024;
    float evaluate(const Board& board) const {
        const bool whiteToMove = (board.sideToMove() == Color::WHITE);
        const Color stm = board.sideToMove();

        float bufA[MAXW], bufB[MAXW];
        float* cur = bufA;
        float* nxt = bufB;

        // ---- layer 0, sparse over the pieces on the board ----
        const Layer& L0 = layers[0];
        for (int j = 0; j < L0.out; j++) cur[j] = L0.b[j];
        for (int sq = 0; sq < 64; sq++) {
            Piece p = board.at(Square(sq));
            if (p == Piece::NONE) continue;
            int idx = pieceIndex(p.type());
            bool own = (p.color() == stm);
            int plane = own ? idx : idx + 6;
            int sqp = whiteToMove ? sq : (sq ^ 56);
            const float* row = &L0.W[(size_t)(plane * 64 + sqp) * L0.out];
            for (int j = 0; j < L0.out; j++) cur[j] += row[j];
        }
        int curSize = L0.out;

        // ---- remaining layers, dense ----
        for (size_t l = 1; l < layers.size(); l++) {
            const Layer& L = layers[l];
            for (int i = 0; i < curSize; i++) if (cur[i] < 0.0f) cur[i] = 0.0f; // ReLU
            for (int j = 0; j < L.out; j++) nxt[j] = L.b[j];
            for (int i = 0; i < L.in; i++) {
                float ai = cur[i];
                if (ai == 0.0f) continue;
                const float* row = &L.W[(size_t)i * L.out];
                for (int j = 0; j < L.out; j++) nxt[j] += ai * row[j];
            }
            float* t = cur; cur = nxt; nxt = t;   // swap buffers
            curSize = L.out;
        }

        float p = 1.0f / (1.0f + expf(-cur[0]));
        const float eps = 1e-6f;
        if (p < eps) p = eps;
        if (p > 1.0f - eps) p = 1.0f - eps;
        return CP_SCALE * logf(p / (1.0f - p));
    }
};

NNUE g_net;

// ===========================================================================
//  SEARCH
// ===========================================================================
static const int MATE = 1000000;          // score for a delivered mate
static const int MATE_THRESH = MATE - 1000; // scores above this are "mate in N"

// ---- transposition table -------------------------------------------------
enum TTFlag : uint8_t { TT_NONE, TT_EXACT, TT_LOWER, TT_UPPER };
struct TTEntry {
    uint64_t key = 0;
    int32_t  score = 0;
    int16_t  depth = -1;
    uint8_t  flag = TT_NONE;
    Move     move = Move::NO_MOVE;
};

class ChessEngine {
public:
    Move bestMove(Board& board, int maxdepth, long long timelimit) {
        starttime = chrono::steady_clock::now();
        timelim = timelimit;
        timesup = false;
        nodes = 0;
        // clear per-search heuristics
        for (auto& k : killers) { k[0] = Move::NO_MOVE; k[1] = Move::NO_MOVE; }
        for (auto& row : history) row.fill(0);

        Movelist moves;
        movegen::legalmoves(moves, board);
        if (moves.empty()) return Move::NO_MOVE;

        Move best = moves[0];
        Move pv = Move::NO_MOVE;

        for (int d = 1; d <= maxdepth; d++) {
            orderMoves(board, moves, pv, 0);
            Move curbest = moves[0];
            int score = -MATE * 2;
            bool broke = false;

            for (const auto& move : moves) {
                board.makeMove(move);
                int newscore = -alphaBeta(board, d - 1, -MATE * 2, MATE * 2, 1);
                board.unmakeMove(move);
                if (timesup) { broke = true; break; }
                if (newscore > score) { score = newscore; curbest = move; }
            }

            if (broke) break;
            best = curbest;
            pv = curbest;

            // UCI info line (nice to watch, and cutechess logs it)
            auto el = chrono::duration_cast<chrono::milliseconds>(
                          chrono::steady_clock::now() - starttime).count();
            cout << "info depth " << d << " score cp " << score
                 << " nodes " << nodes << " time " << el
                 << " pv " << uci::moveToUci(best) << "\n";

            if (score > MATE_THRESH || score < -MATE_THRESH) break; // mate found
            if (timelim > 0 && el >= timelim) break;
        }
        return best;
    }

private:
    chrono::steady_clock::time_point starttime;
    long long timelim = -1;
    bool timesup = false;
    long long nodes = 0;

    // heuristics
    static const int MAXPLY = 128;
    array<array<Move, 2>, MAXPLY> killers;
    array<array<int, 64>, 64> history;   // history[from][to]

    // transposition table (power-of-two size for cheap masking)
    static const size_t TT_BITS = 22;               // ~4M entries
    static const size_t TT_SIZE = (size_t)1 << TT_BITS;
    static const size_t TT_MASK = TT_SIZE - 1;
    vector<TTEntry> tt = vector<TTEntry>(TT_SIZE);

    int pieceVal(PieceType type) {
        if (type == PieceType::PAWN)   return 100;
        if (type == PieceType::KNIGHT) return 320;
        if (type == PieceType::BISHOP) return 330;
        if (type == PieceType::ROOK)   return 500;
        if (type == PieceType::QUEEN)  return 900;
        return 0;
    }

    bool checkTime() {
        if ((++nodes % 2048) == 0 && timelim > 0) {
            auto el = chrono::duration_cast<chrono::milliseconds>(
                          chrono::steady_clock::now() - starttime).count();
            if (el >= timelim) timesup = true;
        }
        return timesup;
    }

    // score a move for ordering (higher = try earlier)
    int moveScore(Board& board, Move mv, Move ttMove, int ply) {
        if (mv == ttMove) return 2000000;
        if (board.isCapture(mv)) {
            int vic = pieceVal(board.at(mv.to()).type());
            int atk = pieceVal(board.at(mv.from()).type());
            return 1000000 + vic * 100 - atk;   // MVV-LVA
        }
        if (mv == killers[ply][0]) return 900000;
        if (mv == killers[ply][1]) return 800000;
        return history[mv.from().index()][mv.to().index()];
    }

    void orderMoves(Board& board, Movelist& moves, Move ttMove, int ply) {
        int n = moves.size();
        vector<int> sc(n);
        for (int i = 0; i < n; i++) sc[i] = moveScore(board, moves[i], ttMove, ply);
        // selection sort (move lists are short)
        for (int i = 0; i < n; i++) {
            int bestj = i;
            for (int j = i + 1; j < n; j++) if (sc[j] > sc[bestj]) bestj = j;
            if (bestj != i) { swap(sc[i], sc[bestj]); swap(moves[i], moves[bestj]); }
        }
    }

    // ---- quiescence: only captures (plus full moves when in check) ---------
    int quiesce(Board& board, int alpha, int beta, int ply) {
        if (checkTime()) return 0;

        bool inChk = board.inCheck();
        int standPat = -MATE * 2;
        if (!inChk) {
            standPat = (int)lround(g_net.evaluate(board));
            if (standPat >= beta) return standPat;      // fail-high, stop
            if (standPat > alpha) alpha = standPat;
        }

        Movelist moves;
        if (inChk) movegen::legalmoves(moves, board);   // must consider all escapes
        else       movegen::legalmoves<movegen::MoveGenType::CAPTURE>(moves, board);

        if (moves.empty()) {
            if (inChk) return -MATE + ply;               // checkmate
            return standPat;                             // quiet, no captures
        }

        orderMoves(board, moves, Move::NO_MOVE, ply);
        int best = standPat;
        for (const auto& move : moves) {
            board.makeMove(move);
            int score = -quiesce(board, -beta, -alpha, ply + 1);
            board.unmakeMove(move);
            if (timesup) return best > -MATE * 2 ? best : 0;
            if (score > best) best = score;
            if (score > alpha) alpha = score;
            if (alpha >= beta) break;
        }
        return best;
    }

    int alphaBeta(Board& board, int depth, int alpha, int beta, int ply) {
        if (checkTime()) return 0;

        // draw by repetition / 50-move (not at root)
        if (ply > 0 && (board.isRepetition(1) || board.isHalfMoveDraw()))
            return 0;

        int alphaOrig = alpha;

        // ---- TT probe ----
        TTEntry& e = tt[board.hash() & TT_MASK];
        Move ttMove = Move::NO_MOVE;
        if (e.key == board.hash()) {
            ttMove = e.move;
            if (e.depth >= depth) {
                if (e.flag == TT_EXACT) return e.score;
                if (e.flag == TT_LOWER && e.score > alpha) alpha = e.score;
                else if (e.flag == TT_UPPER && e.score < beta) beta = e.score;
                if (alpha >= beta) return e.score;
            }
        }

        // check extension: searching a check deeper is cheap and important
        bool inChk = board.inCheck();
        if (inChk) depth++;

        if (depth <= 0) return quiesce(board, alpha, beta, ply);

        Movelist moves;
        movegen::legalmoves(moves, board);
        if (moves.empty())
            return inChk ? -MATE + ply : 0;              // mate or stalemate

        orderMoves(board, moves, ttMove, ply);

        int best = -MATE * 2;
        Move bestMove = moves[0];
        for (int i = 0; i < moves.size(); i++) {
            Move move = moves[i];
            board.makeMove(move);
            int score = -alphaBeta(board, depth - 1, -beta, -alpha, ply + 1);
            board.unmakeMove(move);
            if (timesup) return best > -MATE * 2 ? best : 0;

            if (score > best) { best = score; bestMove = move; }
            if (score > alpha) alpha = score;
            if (alpha >= beta) {
                // beta cutoff: reward this quiet move for next time
                if (!board.isCapture(move)) {
                    if (!(move == killers[ply][0])) {
                        killers[ply][1] = killers[ply][0];
                        killers[ply][0] = move;
                    }
                    history[move.from().index()][move.to().index()] += depth * depth;
                }
                break;
            }
        }

        // ---- TT store ----
        uint8_t flag = (best <= alphaOrig) ? TT_UPPER
                     : (best >= beta)      ? TT_LOWER : TT_EXACT;
        if (depth >= e.depth || e.key != board.hash()) {
            e.key = board.hash(); e.score = best; e.depth = (int16_t)depth;
            e.flag = flag; e.move = bestMove;
        }
        return best;
    }
};

Board board;
ChessEngine engine;

void parsePosition(const string& line) {
    stringstream ss(line);
    string token; ss >> token; ss >> token;
    if (token == "startpos") board.setFen(constants::STARTPOS);
    else if (token == "fen") {
        string fen;
        for (int i = 0; i < 6; i++) { ss >> token; fen += token; if (i != 5) fen += " "; }
        board.setFen(fen);
    }
    if (ss >> token && token == "moves")
        while (ss >> token) board.makeMove(uci::uciToMove(board, token));
}

int parsedepth = -1;
long long parsemovetime = -1, parsewtime = -1, parsebtime = -1, parsewinc = 0, parsebinc = 0;
int parsemtg = -1;

void parseGo(const string& line) {
    stringstream ss(line); string token; ss >> token;
    parsedepth = -1; parsemovetime = -1; parsewtime = -1; parsebtime = -1;
    parsewinc = 0; parsebinc = 0; parsemtg = -1;
    while (ss >> token) {
        if (token == "depth") ss >> parsedepth;
        else if (token == "movetime") ss >> parsemovetime;
        else if (token == "wtime") ss >> parsewtime;
        else if (token == "btime") ss >> parsebtime;
        else if (token == "winc") ss >> parsewinc;
        else if (token == "binc") ss >> parsebinc;
        else if (token == "movestogo") ss >> parsemtg;
    }
}

int main(int argc, char** argv) {
    string wpath = (argc > 1) ? argv[1] : "weights.txt";
    if (g_net.load(wpath)) cout << "info string NN weights loaded from " << wpath << "\n";
    else cout << "info string NN weights NOT loaded (" << wpath << ")\n";

    board.setFen(constants::STARTPOS);
    string line;
    cout.setf(ios::unitbuf);
    while (getline(cin, line)) {
        if (line == "uci") cout << "id name RajitBot-v2\nid author Rajit\nuciok\n";
        else if (line == "isready") cout << "readyok\n";
        else if (line == "ucinewgame") board.setFen(constants::STARTPOS);
        else if (line.rfind("position", 0) == 0) parsePosition(line);
        else if (line.rfind("go", 0) == 0) {
            parseGo(line);
            int usedepth = 64; long long usetime = -1;
            if (parsedepth > 0) { usedepth = parsedepth; usetime = -1; }
            else if (parsemovetime > 0) usetime = parsemovetime;
            else {
                bool white = board.sideToMove() == Color::WHITE;
                long long mytime = white ? parsewtime : parsebtime;
                long long myinc = white ? parsewinc : parsebinc;
                if (mytime > 0) {
                    int mtg = parsemtg > 0 ? parsemtg : 30;
                    usetime = mytime / mtg + myinc / 2 - 50;
                    if (usetime < 50) usetime = 50;
                } else { usedepth = 6; usetime = -1; }
            }
            Move best = engine.bestMove(board, usedepth, usetime);
            cout << "bestmove " << uci::moveToUci(best) << endl;
        }
        else if (line == "quit") break;
    }
}
