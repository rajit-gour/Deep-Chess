// engine_v3: engine_v2 + modern search + a strong fast classical eval.
//
// Search additions over v2:
//   * PRINCIPAL VARIATION SEARCH (PVS)  -- first move full window, rest with a
//     null window and re-search only if they beat alpha.
//   * NULL-MOVE PRUNING                 -- "if I skip my move and I'm still
//     winning, this node is too good; prune." Big speedup in non-endgames.
//   * LATE-MOVE REDUCTIONS (LMR)        -- search late, quiet moves shallower;
//     re-search at full depth only if they surprise us.
//   * ASPIRATION WINDOWS                -- re-search the root inside a narrow
//     window around the last score; widen on failure.
//
// Eval: uses the NN if weights are loaded, otherwise a fast classical eval
//   (material + piece-square tables, centipawns). So:
//     engine_v3.exe weights_base200.txt   -> NN bot
//     engine_v3.exe                        -> strong classical bot
//
// BUILD: g++ -std=c++17 -O3 -static engine_v3.cpp -o engine_v3.exe

#include <fstream>
#include <iostream>
#include <sstream>
#include "chess.hpp"
#include <cmath>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <array>

using namespace chess;
using namespace std;

// ===========================================================================
//  NEURAL NETWORK  (fast, no-alloc; identical math to train.py)
// ===========================================================================
struct NNUE {
    static const int IN = 768;
    static constexpr float CP_SCALE = 400.0f;
    static const int MAXW = 1024;
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
        if (t == PieceType::PAWN) return 0; if (t == PieceType::KNIGHT) return 1;
        if (t == PieceType::BISHOP) return 2; if (t == PieceType::ROOK) return 3;
        if (t == PieceType::QUEEN) return 4; return 5;
    }
    float evaluate(const Board& board) const {
        const bool wtm = (board.sideToMove() == Color::WHITE);
        const Color stm = board.sideToMove();
        float bufA[MAXW], bufB[MAXW]; float* cur = bufA; float* nxt = bufB;
        const Layer& L0 = layers[0];
        for (int j = 0; j < L0.out; j++) cur[j] = L0.b[j];
        for (int sq = 0; sq < 64; sq++) {
            Piece p = board.at(Square(sq));
            if (p == Piece::NONE) continue;
            int idx = pieceIndex(p.type());
            bool own = (p.color() == stm);
            int plane = own ? idx : idx + 6;
            int sqp = wtm ? sq : (sq ^ 56);
            const float* row = &L0.W[(size_t)(plane * 64 + sqp) * L0.out];
            for (int j = 0; j < L0.out; j++) cur[j] += row[j];
        }
        int curSize = L0.out;
        for (size_t l = 1; l < layers.size(); l++) {
            const Layer& L = layers[l];
            for (int i = 0; i < curSize; i++) if (cur[i] < 0) cur[i] = 0;
            for (int j = 0; j < L.out; j++) nxt[j] = L.b[j];
            for (int i = 0; i < L.in; i++) {
                float ai = cur[i]; if (ai == 0) continue;
                const float* row = &L.W[(size_t)i * L.out];
                for (int j = 0; j < L.out; j++) nxt[j] += ai * row[j];
            }
            float* t = cur; cur = nxt; nxt = t; curSize = L.out;
        }
        float p = 1.0f / (1.0f + expf(-cur[0]));
        const float e = 1e-6f; if (p < e) p = e; if (p > 1 - e) p = 1 - e;
        return CP_SCALE * logf(p / (1 - p));
    }
};
NNUE g_net;

// ===========================================================================
//  CLASSICAL EVAL (fast fallback): material + piece-square tables (centipawns)
// ===========================================================================
namespace classical {
// values in cp
static const int PVAL[6] = {100, 320, 330, 500, 900, 0};
// piece-square tables, written in reading order (index 0 = a8 ... 63 = h1),
// from White's point of view. Simplified Michniewski tables.
static const int PST[6][64] = {
  { // pawn
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0 },
  { // knight
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50 },
  { // bishop
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20 },
  { // rook
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0 },
  { // queen
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20 },
  { // king (midgame)
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20 }
};
static int idx(PieceType t) {
    if (t == PieceType::PAWN) return 0; if (t == PieceType::KNIGHT) return 1;
    if (t == PieceType::BISHOP) return 2; if (t == PieceType::ROOK) return 3;
    if (t == PieceType::QUEEN) return 4; return 5;
}
// white-relative, then flipped to side-to-move by caller
int evaluate(const Board& board) {
    int sc = 0;
    for (int sq = 0; sq < 64; sq++) {
        Piece p = board.at(Square(sq));
        if (p == Piece::NONE) continue;
        int i = idx(p.type());
        bool white = (p.color() == Color::WHITE);
        int pst = PST[i][white ? (sq ^ 56) : sq];
        int v = PVAL[i] + pst;
        sc += white ? v : -v;
    }
    return board.sideToMove() == Color::WHITE ? sc : -sc;
}
} // namespace classical

static inline int evalPos(const Board& b) {
    return g_net.loaded ? (int)lround(g_net.evaluate(b)) : classical::evaluate(b);
}

// ===========================================================================
//  SEARCH
// ===========================================================================
static const int MATE = 1000000, MATE_THRESH = MATE - 1000, INF = MATE * 2;
enum TTFlag : uint8_t { TT_NONE, TT_EXACT, TT_LOWER, TT_UPPER };
struct TTEntry { uint64_t key = 0; int32_t score = 0; int16_t depth = -1; uint8_t flag = TT_NONE; Move move = Move::NO_MOVE; };

class ChessEngine {
public:
    Move bestMove(Board& board, int maxdepth, long long timelimit) {
        starttime = chrono::steady_clock::now();
        timelim = timelimit; timesup = false; nodes = 0;
        for (auto& k : killers) { k[0] = Move::NO_MOVE; k[1] = Move::NO_MOVE; }
        for (auto& row : history) row.fill(0);

        Movelist moves; movegen::legalmoves(moves, board);
        if (moves.empty()) return Move::NO_MOVE;
        Move best = moves[0];
        int prevScore = 0;

        for (int d = 1; d <= maxdepth; d++) {
            // aspiration window around the previous iteration's score
            int alpha = -INF, beta = INF, delta = 30;
            if (d >= 4) { alpha = prevScore - delta; beta = prevScore + delta; }
            int score;
            while (true) {
                score = searchRoot(board, d, alpha, beta, best);
                if (timesup) break;
                if (score <= alpha) { alpha = max(-INF, alpha - delta * 2); delta *= 2; }
                else if (score >= beta) { beta = min(INF, beta + delta * 2); delta *= 2; }
                else break;
            }
            if (timesup) break;
            prevScore = score;
            auto el = ms();
            cout << "info depth " << d << " score cp " << score
                 << " nodes " << nodes << " time " << el
                 << " pv " << uci::moveToUci(best) << "\n";
            if (score > MATE_THRESH || score < -MATE_THRESH) break;
            if (timelim > 0 && el >= timelim) break;
        }
        return best;
    }

private:
    chrono::steady_clock::time_point starttime;
    long long timelim = -1, nodes = 0;
    bool timesup = false;
    static const int MAXPLY = 128;
    array<array<Move, 2>, MAXPLY> killers;
    array<array<int, 64>, 64> history;
    static const size_t TT_BITS = 22, TT_SIZE = (size_t)1 << TT_BITS, TT_MASK = TT_SIZE - 1;
    vector<TTEntry> tt = vector<TTEntry>(TT_SIZE);

    long long ms() { return chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - starttime).count(); }
    bool checkTime() { if ((++nodes % 2048) == 0 && timelim > 0 && ms() >= timelim) timesup = true; return timesup; }

    int pieceVal(PieceType t) {
        if (t == PieceType::PAWN) return 100; if (t == PieceType::KNIGHT) return 320;
        if (t == PieceType::BISHOP) return 330; if (t == PieceType::ROOK) return 500;
        if (t == PieceType::QUEEN) return 900; return 0;
    }
    int moveScore(Board& b, Move mv, Move ttMove, int ply) {
        if (mv == ttMove) return 2000000;
        if (b.isCapture(mv)) return 1000000 + pieceVal(b.at(mv.to()).type()) * 100 - pieceVal(b.at(mv.from()).type());
        if (mv == killers[ply][0]) return 900000;
        if (mv == killers[ply][1]) return 800000;
        return history[mv.from().index()][mv.to().index()];
    }
    void orderMoves(Board& b, Movelist& m, Move ttMove, int ply) {
        int n = m.size(); int sc[256];
        for (int i = 0; i < n; i++) sc[i] = moveScore(b, m[i], ttMove, ply);
        for (int i = 0; i < n; i++) { int bj = i; for (int j = i + 1; j < n; j++) if (sc[j] > sc[bj]) bj = j;
            if (bj != i) { swap(sc[i], sc[bj]); swap(m[i], m[bj]); } }
    }
    bool hasBigPieces(const Board& b, Color c) {
        return (bool)b.pieces(PieceType::KNIGHT, c) || (bool)b.pieces(PieceType::BISHOP, c)
            || (bool)b.pieces(PieceType::ROOK, c) || (bool)b.pieces(PieceType::QUEEN, c);
    }

    int quiesce(Board& b, int alpha, int beta, int ply) {
        if (checkTime()) return 0;
        bool inChk = b.inCheck();
        int standPat = -INF;
        if (!inChk) { standPat = evalPos(b); if (standPat >= beta) return standPat; if (standPat > alpha) alpha = standPat; }
        Movelist moves;
        if (inChk) movegen::legalmoves(moves, b);
        else movegen::legalmoves<movegen::MoveGenType::CAPTURE>(moves, b);
        if (moves.empty()) return inChk ? -MATE + ply : standPat;
        orderMoves(b, moves, Move::NO_MOVE, ply);
        int best = standPat;
        for (const auto& mv : moves) {
            b.makeMove(mv); int s = -quiesce(b, -beta, -alpha, ply + 1); b.unmakeMove(mv);
            if (timesup) return best > -INF ? best : 0;
            if (s > best) best = s; if (s > alpha) alpha = s; if (alpha >= beta) break;
        }
        return best;
    }

    // root: like alphaBeta but tracks the best move
    int searchRoot(Board& b, int depth, int alpha, int beta, Move& best) {
        Movelist moves; movegen::legalmoves(moves, b);
        orderMoves(b, moves, best, 0);
        int bestScore = -INF; bool first = true;
        for (const auto& mv : moves) {
            b.makeMove(mv);
            int s;
            if (first) s = -alphaBeta(b, depth - 1, -beta, -alpha, 1, true);
            else {
                s = -alphaBeta(b, depth - 1, -alpha - 1, -alpha, 1, true);
                if (s > alpha && s < beta) s = -alphaBeta(b, depth - 1, -beta, -alpha, 1, true);
            }
            b.unmakeMove(mv);
            if (timesup) break;
            if (s > bestScore) { bestScore = s; best = mv; }
            if (s > alpha) alpha = s;
            if (alpha >= beta) break;
            first = false;
        }
        return bestScore;
    }

    int alphaBeta(Board& b, int depth, int alpha, int beta, int ply, bool canNull) {
        if (checkTime()) return 0;
        if (ply > 0 && (b.isRepetition(1) || b.isHalfMoveDraw())) return 0;
        int alphaOrig = alpha;
        bool isPV = (beta - alpha) > 1;

        TTEntry& e = tt[b.hash() & TT_MASK];
        Move ttMove = Move::NO_MOVE;
        if (e.key == b.hash()) {
            ttMove = e.move;
            if (e.depth >= depth && !isPV) {
                if (e.flag == TT_EXACT) return e.score;
                if (e.flag == TT_LOWER && e.score >= beta) return e.score;
                if (e.flag == TT_UPPER && e.score <= alpha) return e.score;
            }
        }

        bool inChk = b.inCheck();
        if (inChk) depth++;                       // check extension
        if (depth <= 0) return quiesce(b, alpha, beta, ply);

        // null-move pruning
        if (canNull && !isPV && !inChk && depth >= 3 &&
            hasBigPieces(b, b.sideToMove()) && beta < MATE_THRESH) {
            int R = 2 + depth / 6;
            b.makeNullMove();
            int s = -alphaBeta(b, depth - 1 - R, -beta, -beta + 1, ply + 1, false);
            b.unmakeNullMove();
            if (timesup) return 0;
            if (s >= beta) return beta;
        }

        Movelist moves; movegen::legalmoves(moves, b);
        if (moves.empty()) return inChk ? -MATE + ply : 0;
        orderMoves(b, moves, ttMove, ply);

        int best = -INF; Move bestMove = moves[0];
        int searched = 0;
        for (int i = 0; i < moves.size(); i++) {
            Move mv = moves[i];
            bool isCap = b.isCapture(mv);
            bool givesChk = b.givesCheck(mv) != CheckType::NO_CHECK;
            b.makeMove(mv);

            int s;
            if (searched == 0) {
                s = -alphaBeta(b, depth - 1, -beta, -alpha, ply + 1, true);
            } else {
                // late move reduction for quiet, non-checking, later moves
                int red = 0;
                if (depth >= 3 && searched >= 3 && !isCap && !givesChk && !inChk) {
                    red = 1 + (searched >= 6 ? 1 : 0);
                }
                s = -alphaBeta(b, depth - 1 - red, -alpha - 1, -alpha, ply + 1, true);
                if (s > alpha && red > 0)          // reduced search surprised us
                    s = -alphaBeta(b, depth - 1, -alpha - 1, -alpha, ply + 1, true);
                if (s > alpha && s < beta)         // PVS re-search full window
                    s = -alphaBeta(b, depth - 1, -beta, -alpha, ply + 1, true);
            }
            b.unmakeMove(mv);
            if (timesup) return best > -INF ? best : 0;

            searched++;
            if (s > best) { best = s; bestMove = mv; }
            if (s > alpha) alpha = s;
            if (alpha >= beta) {
                if (!isCap) {
                    if (!(mv == killers[ply][0])) { killers[ply][1] = killers[ply][0]; killers[ply][0] = mv; }
                    history[mv.from().index()][mv.to().index()] += depth * depth;
                }
                break;
            }
        }

        uint8_t flag = (best <= alphaOrig) ? TT_UPPER : (best >= beta) ? TT_LOWER : TT_EXACT;
        if (depth >= e.depth || e.key != b.hash()) {
            e.key = b.hash(); e.score = best; e.depth = (int16_t)depth; e.flag = flag; e.move = bestMove;
        }
        return best;
    }
};

Board board; ChessEngine engine;

void parsePosition(const string& line) {
    stringstream ss(line); string t; ss >> t; ss >> t;
    if (t == "startpos") board.setFen(constants::STARTPOS);
    else if (t == "fen") { string fen; for (int i = 0; i < 6; i++) { ss >> t; fen += t; if (i != 5) fen += " "; } board.setFen(fen); }
    if (ss >> t && t == "moves") while (ss >> t) board.makeMove(uci::uciToMove(board, t));
}
int pd = -1; long long pmt = -1, pwt = -1, pbt = -1, pwi = 0, pbi = 0; int pmtg = -1;
void parseGo(const string& line) {
    stringstream ss(line); string t; ss >> t;
    pd = -1; pmt = -1; pwt = -1; pbt = -1; pwi = 0; pbi = 0; pmtg = -1;
    while (ss >> t) {
        if (t == "depth") ss >> pd; else if (t == "movetime") ss >> pmt;
        else if (t == "wtime") ss >> pwt; else if (t == "btime") ss >> pbt;
        else if (t == "winc") ss >> pwi; else if (t == "binc") ss >> pbi;
        else if (t == "movestogo") ss >> pmtg;
    }
}
int main(int argc, char** argv) {
    string wpath = (argc > 1) ? argv[1] : "weights.txt";
    if (g_net.load(wpath)) cout << "info string NN eval (" << wpath << ")\n";
    else cout << "info string classical eval (no weights)\n";
    board.setFen(constants::STARTPOS);
    string line; cout.setf(ios::unitbuf);
    while (getline(cin, line)) {
        if (line == "uci") cout << "id name RajitBot-v3\nid author Rajit\nuciok\n";
        else if (line == "isready") cout << "readyok\n";
        else if (line == "ucinewgame") board.setFen(constants::STARTPOS);
        else if (line.rfind("position", 0) == 0) parsePosition(line);
        else if (line.rfind("go", 0) == 0) {
            parseGo(line);
            int ud = 64; long long ut = -1;
            if (pd > 0) { ud = pd; ut = -1; }
            else if (pmt > 0) ut = pmt;
            else {
                bool white = board.sideToMove() == Color::WHITE;
                long long myt = white ? pwt : pbt, myi = white ? pwi : pbi;
                if (myt > 0) { int mtg = pmtg > 0 ? pmtg : 30; ut = myt / mtg + myi / 2 - 50; if (ut < 50) ut = 50; }
                else { ud = 8; ut = -1; }
            }
            Move best = engine.bestMove(board, ud, ut);
            cout << "bestmove " << uci::moveToUci(best) << endl;
        }
        else if (line == "quit") break;
    }
}
