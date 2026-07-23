// NN-evaluated chess engine (week6).
//
// This is week5.cpp with ONE change: evaluate() now runs a neural network
// instead of the material+mobility heuristic. Search, move ordering, time
// management and UCI are unchanged.
//
// The network (768 -> 256 -> 1) is trained offline in Python (train.py) and
// its weights are loaded from weights.txt at startup. If weights.txt is
// missing, we fall back to the old material eval so the engine still runs.
//
// BUILD:  g++ -std=c++17 -O3 -static engine_nn.cpp -o engine_nn.exe
//         (put weights.txt next to the exe, or pass its path as argv[1])

#include <fstream>
#include <iostream>
#include <sstream>
#include "chess.hpp"
#include <cmath>
#include <chrono>
#include <algorithm>
#include <vector>

using namespace chess;
using namespace std;

// ===========================================================================
//  NEURAL NETWORK  (inference only)
// ===========================================================================
// Layout MUST match train.py exactly:
//   IN=768, H=256, OUT=1
//   W1 stored row-major as (IN x H):  W1[i*H + j]
//   W2 stored row-major as (H  x 1):  W2[j]
//   feature index = plane*64 + (whiteToMove ? sq : sq^56)
//   planes 0..5  = OWN   P,N,B,R,Q,K   (own = side to move)
//   planes 6..11 = ENEMY P,N,B,R,Q,K
struct NNUE {
    static const int IN = 768;                   // must equal layer 0 input
    static constexpr float CP_SCALE = 400.0f;    // same constant as train.py

    // one fully-connected layer: out = W @ in + b, W stored row-major W[i*out+j]
    struct Layer { int in, out; vector<float> W, b; };
    vector<Layer> layers;
    bool loaded = false;

    // Reads either format:
    //   generic:      "LAYERS N", then per layer: "in out", W(in*out), b(out)
    //   legacy 1-hid: "768 256 1", then W1, b1, W2, b2   (from train.py)
    bool load(const string& path) {
        ifstream f(path);
        if (!f) return false;
        string tag;
        f >> tag;
        if (tag == "LAYERS") {
            int n; f >> n;
            layers.resize(n);
            for (auto& L : layers) {
                f >> L.in >> L.out;
                L.W.resize((size_t)L.in * L.out); L.b.resize(L.out);
                for (auto& v : L.W) f >> v;
                for (auto& v : L.b) f >> v;
            }
        } else {
            // legacy: tag holds the first int (IN); read H and OUT
            int in = stoi(tag), h, out;
            f >> h >> out;
            layers.resize(2);
            layers[0] = {in, h, vector<float>((size_t)in * h), vector<float>(h)};
            layers[1] = {h, out, vector<float>((size_t)h * out), vector<float>(out)};
            for (auto& v : layers[0].W) f >> v;   // W1
            for (auto& v : layers[0].b) f >> v;   // b1
            for (auto& v : layers[1].W) f >> v;   // W2
            for (auto& v : layers[1].b) f >> v;   // b2
        }
        loaded = (bool)f && !layers.empty() && layers.front().in == IN;
        return loaded;
    }

    // piece type -> 0..5 in the order P,N,B,R,Q,K (matches train.py PIECE_ORDER)
    static int pieceIndex(PieceType t) {
        if (t == PieceType::PAWN)   return 0;
        if (t == PieceType::KNIGHT) return 1;
        if (t == PieceType::BISHOP) return 2;
        if (t == PieceType::ROOK)   return 3;
        if (t == PieceType::QUEEN)  return 4;
        return 5; // KING
    }

    // Forward pass -> centipawns from the side-to-move's perspective.
    float evaluate(const Board& board) const {
        const bool whiteToMove = (board.sideToMove() == Color::WHITE);
        const Color stm = board.sideToMove();

        // ---- layer 0, computed sparsely -----------------------------------
        // The 768 inputs are binary and only ~32 are 1, so instead of a full
        // matrix multiply we just ADD the W rows of the pieces on the board.
        // (This is the idea NNUE makes "efficiently updatable" later.)
        const Layer& L0 = layers[0];
        vector<float> a = L0.b;                    // start at the bias
        for (int sq = 0; sq < 64; sq++) {
            Piece p = board.at(Square(sq));
            if (p == Piece::NONE) continue;
            int idx   = pieceIndex(p.type());
            bool own  = (p.color() == stm);
            int plane = own ? idx : idx + 6;
            int sqp   = whiteToMove ? sq : (sq ^ 56);
            const float* row = &L0.W[(size_t)(plane * 64 + sqp) * L0.out];
            for (int j = 0; j < L0.out; j++) a[j] += row[j];
        }

        // ---- remaining layers, dense --------------------------------------
        for (size_t l = 1; l < layers.size(); l++) {
            const Layer& L = layers[l];
            // ReLU on the input to this layer (activation of the previous one)
            for (float& v : a) if (v < 0.0f) v = 0.0f;
            vector<float> z = L.b;
            for (int i = 0; i < L.in; i++) {
                float ai = a[i];
                if (ai == 0.0f) continue;
                const float* row = &L.W[(size_t)i * L.out];
                for (int j = 0; j < L.out; j++) z[j] += ai * row[j];
            }
            a.swap(z);
        }

        // final layer output is raw; sigmoid -> win prob -> centipawns
        float p = 1.0f / (1.0f + expf(-a[0]));
        const float eps = 1e-6f;
        if (p < eps) p = eps;
        if (p > 1.0f - eps) p = 1.0f - eps;
        return CP_SCALE * logf(p / (1.0f - p));
    }
};

NNUE g_net;

// ===========================================================================
//  ENGINE  (identical to week5 except evaluate())
// ===========================================================================
class ChessEngine{
public:
    Move bestMove(Board& board, int maxdepth, long long timelimit) {
        starttime=chrono::steady_clock::now();
        timelim=timelimit;
        timesup=false;
        nodes=0;

        Movelist moves;
        movegen::legalmoves(moves,board);
        if (moves.empty()){
            return Move::NO_MOVE;
        }

        Move best=moves[0];
        Move pv=Move::NO_MOVE;

        for (int d=1; d<=maxdepth; d++){
            orderMoves(board,moves,pv);
            Move curbest=moves[0];
            float score=-INFINITY;
            bool broke=false;

            for (const auto& move : moves){
                board.makeMove(move);
                float newscore=-alphaBeta(board,d-1,-INFINITY,INFINITY,1);
                board.unmakeMove(move);

                if (timesup){
                    broke=true;
                    break;
                }

                if (newscore>score){
                    score=newscore;
                    curbest=move;
                }
            }

            if (broke) break;
            best=curbest;
            pv=curbest;

            auto elapsed=chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now()-starttime).count();
            if (timelim>0 && elapsed>=timelim) break;
        }
        return best;
    }
private:
    chrono::steady_clock::time_point starttime;
    long long timelim=-1;
    bool timesup=false;
    long long nodes=0;

    float pieceVal(PieceType type){
        if (type==PieceType::PAWN) return 1;
        if (type==PieceType::KNIGHT) return 3;
        if (type==PieceType::BISHOP) return 3;
        if (type==PieceType::ROOK) return 5;
        if (type==PieceType::QUEEN) return 9;
        return 0;
    }

    void orderMoves(Board& board, Movelist& moves, Move pv){
        vector<float> scores(moves.size());
        for (int i=0; i<moves.size(); i++){
            Move mv=moves[i];
            float sc=0;
            if (mv==pv){
                sc=99999;
            }
            else if (board.isCapture(mv)){
                float vic=pieceVal(board.at(mv.to()).type());
                float atk=pieceVal(board.at(mv.from()).type());
                sc=1000+vic*10-atk;
            }
            scores[i]=sc;
        }
        for (int i=0; i<moves.size(); i++){
            for (int j=i+1; j<moves.size(); j++){
                if (scores[j]>scores[i]){
                    float tmp=scores[i]; scores[i]=scores[j]; scores[j]=tmp;
                    Move tmpm=moves[i]; moves[i]=moves[j]; moves[j]=tmpm;
                }
            }
        }
    }

    // ---- the ONE changed function ------------------------------------------
    // Old: material + 0.1*mobility.  New: neural-network forward pass.
    // Returns a score in centipawns, relative to the side to move (positive =
    // good for the player about to move) -- same convention as before, so the
    // negamax search in alphaBeta() needs no changes. Mate scores in alphaBeta
    // (+-10000) still dwarf any NN eval (bounded to ~ +-5500 cp).
    float evaluate(Board& board) {
        if (g_net.loaded)
            return g_net.evaluate(board);

        // fallback: original hand-crafted eval (kept so the engine runs even
        // without weights.txt). Note: returns pawns, not centipawns.
        float score = 0;
        Movelist mobilityMoves;
        movegen::legalmoves(mobilityMoves, board);
        score += 0.1 * mobilityMoves.size();
        for (int sq = 0; sq < 64; sq++) {
            Piece p = board.at(Square(sq));
            if (p == Piece::NONE) continue;
            float val = pieceVal(p.type());
            if (p.color() == board.sideToMove()) score += val;
            else score -= val;
        }
        return score;
    }

    float alphaBeta(Board& board, int depth, float alpha, float beta, int ply){
        nodes++;
        if ((nodes%2048)==0 && timelim>0){
            auto el=chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now()-starttime).count();
            if (el>=timelim) timesup=true;
        }
        if (timesup) return 0;

        Movelist moves;
        movegen::legalmoves(moves,board);

        if (moves.size() ==0){
            if (board.inCheck()){
                return -10000+ply;
            }
            else{
                return 0;
            }
        }
        else if (depth==0){
            return evaluate(board);
        }
        else{
            orderMoves(board,moves,Move::NO_MOVE);
            float hi=-INFINITY;
            for (const auto& move : moves){
                board.makeMove(move);
                float newscore=-alphaBeta(board,depth -1,-beta,-alpha,ply+1);
                board.unmakeMove(move);
                if (timesup) return hi==-INFINITY?0:hi;
                hi=max(hi,newscore);
                alpha=max(alpha,newscore);
                if (beta<=alpha){
                    break;
                }
            }
            return hi;
        }
    }
};

Board board;
ChessEngine engine;

void parsePosition(const string& line) {
    stringstream ss(line);
    string token;
    ss >> token;
    ss >> token;

    if (token == "startpos") {
        board.setFen(constants::STARTPOS);
    } else if (token == "fen") {
        string fen;
        for (int i = 0; i < 6; i++) {
            ss >> token;
            fen += token;
            if (i != 5) fen += " ";
        }
        board.setFen(fen);
    }

    if (ss >> token && token == "moves") {
        while (ss >> token) {
            Move move = uci::uciToMove(board, token);
            board.makeMove(move);
        }
    }
}

int parsedepth=-1;
long long parsemovetime=-1;
long long parsewtime=-1, parsebtime=-1, parsewinc=0, parsebinc=0;
int parsemtg=-1;

void parseGo(const string& line){
    stringstream ss(line);
    string token;
    ss>>token;
    parsedepth=-1; parsemovetime=-1; parsewtime=-1; parsebtime=-1; parsewinc=0; parsebinc=0; parsemtg=-1;
    while (ss>>token){
        if (token=="depth") ss>>parsedepth;
        else if (token=="movetime") ss>>parsemovetime;
        else if (token=="wtime") ss>>parsewtime;
        else if (token=="btime") ss>>parsebtime;
        else if (token=="winc") ss>>parsewinc;
        else if (token=="binc") ss>>parsebinc;
        else if (token=="movestogo") ss>>parsemtg;
    }
}

int main(int argc, char** argv) {
    // load the network (path from argv[1], else weights.txt next to the exe)
    string wpath = (argc > 1) ? argv[1] : "weights.txt";
    if (g_net.load(wpath))
        cout << "info string NN weights loaded from " << wpath << "\n";
    else
        cout << "info string NN weights NOT loaded (" << wpath << "); using material eval\n";

    board.setFen(constants::STARTPOS);

    string line;
    cout.setf(ios::unitbuf);
    while (getline(cin, line)) {
        if (line == "uci") {
            cout << "id name RajitBot-NN\nid author Rajit\nuciok\n";
            cout.flush();
        }
        else if (line == "isready") {
            cout << "readyok\n";
            cout.flush();
        }
        else if (line == "ucinewgame") {
            board.setFen(constants::STARTPOS);
        }
        else if (line.rfind("position", 0) == 0) {
            parsePosition(line);
        }
        else if (line.rfind("go", 0) == 0) {
            parseGo(line);

            int usedepth=64;
            long long usetime=-1;

            if (parsedepth>0){
                usedepth=parsedepth;
                usetime=-1;
            }
            else if (parsemovetime>0){
                usetime=parsemovetime;
            }
            else{
                bool white=board.sideToMove()==Color::WHITE;
                long long mytime=white?parsewtime:parsebtime;
                long long myinc=white?parsewinc:parsebinc;
                if (mytime>0){
                    int mtg=parsemtg>0?parsemtg:30;
                    usetime=mytime/mtg+myinc/2-50;
                    if (usetime<50) usetime=50;
                }
                else{
                    usedepth=5;
                    usetime=-1;
                }
            }

            Move best = engine.bestMove(board,usedepth,usetime);
            cout << "bestmove " << uci::moveToUci(best) << endl;
        }
        else if (line == "quit") {
            break;
        }
    }
}
