//USAGE: g++ -std=c++17 -O3 -static engine_final.cpp -o engine_final.exe
// eval = fast classical (material + piece-square tables) + optional NN correction.
// pass a net file to use the hybrid eval:  engine_final.exe net.txt

#include <fstream>
#include <iostream>
#include <sstream>
#include "chess.hpp"
#include <cmath>
#include <chrono>
#include <algorithm>
#include <vector>
#include <array>

using namespace chess;
using namespace std;

// ---- classical eval: material + piece-square tables (centipawns) ----
namespace classical {
static const int PVAL[6] = {100, 320, 330, 500, 900, 0};
static const int PST[6][64] = {
  { 0,0,0,0,0,0,0,0, 50,50,50,50,50,50,50,50, 10,10,20,30,30,20,10,10,
    5,5,10,25,25,10,5,5, 0,0,0,20,20,0,0,0, 5,-5,-10,0,0,-10,-5,5,
    5,10,10,-20,-20,10,10,5, 0,0,0,0,0,0,0,0 },
  { -50,-40,-30,-30,-30,-30,-40,-50, -40,-20,0,0,0,0,-20,-40, -30,0,10,15,15,10,0,-30,
    -30,5,15,20,20,15,5,-30, -30,0,15,20,20,15,0,-30, -30,5,10,15,15,10,5,-30,
    -40,-20,0,5,5,0,-20,-40, -50,-40,-30,-30,-30,-30,-40,-50 },
  { -20,-10,-10,-10,-10,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,10,10,5,0,-10,
    -10,5,5,10,10,5,5,-10, -10,0,10,10,10,10,0,-10, -10,10,10,10,10,10,10,-10,
    -10,5,0,0,0,0,5,-10, -20,-10,-10,-10,-10,-10,-10,-20 },
  { 0,0,0,0,0,0,0,0, 5,10,10,10,10,10,10,5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5,
    -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, 0,0,0,5,5,0,0,0 },
  { -20,-10,-10,-5,-5,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,5,5,5,0,-10,
    -5,0,5,5,5,5,0,-5, 0,0,5,5,5,5,0,-5, -10,5,5,5,5,5,0,-10,
    -10,0,5,0,0,0,0,-10, -20,-10,-10,-5,-5,-10,-10,-20 },
  { -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30, -20,-30,-30,-40,-40,-30,-30,-20, -10,-20,-20,-20,-20,-20,-20,-10,
    20,20,0,0,0,0,20,20, 20,30,10,0,0,10,30,20 }
};
int idx(PieceType t){
    if (t==PieceType::PAWN) return 0; if (t==PieceType::KNIGHT) return 1;
    if (t==PieceType::BISHOP) return 2; if (t==PieceType::ROOK) return 3;
    if (t==PieceType::QUEEN) return 4; return 5;
}
int evaluate(const Board& board){
    int sc=0;
    for (int sq=0; sq<64; sq++){
        Piece p=board.at(Square(sq));
        if (p==Piece::NONE) continue;
        int i=idx(p.type());
        bool white=(p.color()==Color::WHITE);
        int v=PVAL[i]+PST[i][white?(sq^56):sq];
        sc+=white?v:-v;
    }
    return board.sideToMove()==Color::WHITE?sc:-sc;
}
}

// ---- NN correction: small MLP over the 768 board features, linear output (cp) ----
struct Net {
    struct Layer { int in,out; vector<float> W,b; };
    vector<Layer> layers;
    bool loaded=false;

    bool load(const string& path){
        ifstream f(path); if (!f) return false;
        string tag; int n; f>>tag>>n;      // "LAYERS n"
        if (tag!="LAYERS") return false;
        layers.resize(n);
        for (auto& L:layers){
            f>>L.in>>L.out; L.W.resize((size_t)L.in*L.out); L.b.resize(L.out);
            for (auto& v:L.W) f>>v; for (auto& v:L.b) f>>v;
        }
        loaded=(bool)f; return loaded;
    }
    static int pieceIndex(PieceType t){
        if (t==PieceType::PAWN) return 0; if (t==PieceType::KNIGHT) return 1;
        if (t==PieceType::BISHOP) return 2; if (t==PieceType::ROOK) return 3;
        if (t==PieceType::QUEEN) return 4; return 5;
    }
    // returns a cp correction from the side-to-move's view
    float correction(const Board& board) const {
        const bool wtm=(board.sideToMove()==Color::WHITE);
        const Color stm=board.sideToMove();
        const Layer& L0=layers[0];
        float bufA[1024],bufB[1024]; float* cur=bufA; float* nxt=bufB;
        for (int j=0;j<L0.out;j++) cur[j]=L0.b[j];
        for (int sq=0;sq<64;sq++){           // first layer is sparse over pieces
            Piece p=board.at(Square(sq)); if (p==Piece::NONE) continue;
            int ti=pieceIndex(p.type());
            int plane=(p.color()==stm)?ti:ti+6;
            int sqp=wtm?sq:(sq^56);
            const float* row=&L0.W[(size_t)(plane*64+sqp)*L0.out];
            for (int j=0;j<L0.out;j++) cur[j]+=row[j];
        }
        int sz=L0.out;
        for (size_t l=1;l<layers.size();l++){
            const Layer& L=layers[l];
            for (int i=0;i<sz;i++) if (cur[i]<0) cur[i]=0;   // ReLU
            for (int j=0;j<L.out;j++) nxt[j]=L.b[j];
            for (int i=0;i<L.in;i++){
                float a=cur[i]; if (a==0) continue;
                const float* row=&L.W[(size_t)i*L.out];
                for (int j=0;j<L.out;j++) nxt[j]+=a*row[j];
            }
            float* t=cur; cur=nxt; nxt=t; sz=L.out;
        }
        return cur[0];                        // linear output = cp correction
    }
};
Net g_net;

int evalPos(const Board& b){
    int e=classical::evaluate(b);
    if (g_net.loaded) e+=(int)lround(g_net.correction(b));
    return e;
}

// ---- search ----
static const int MATE=1000000, MATE_TH=MATE-1000, INF=MATE*2;
enum { TT_NONE, TT_EXACT, TT_LOWER, TT_UPPER };
struct TTEntry { uint64_t key=0; int score=0; short depth=-1; unsigned char flag=TT_NONE; Move move=Move::NO_MOVE; };

class ChessEngine {
public:
    Move bestMove(Board& board, int maxdepth, long long timelimit){
        start=chrono::steady_clock::now(); tlim=timelimit; stop=false; nodes=0;
        for (auto& k:killers){ k[0]=Move::NO_MOVE; k[1]=Move::NO_MOVE; }
        for (auto& r:history) r.fill(0);

        Movelist moves; movegen::legalmoves(moves,board);
        if (moves.empty()) return Move::NO_MOVE;
        Move best=moves[0];

        for (int d=1; d<=maxdepth; d++){
            Move bm=best;
            int sc=searchRoot(board,d,bm);
            if (stop) break;
            best=bm;
            long long el=ms();
            cout<<"info depth "<<d<<" score cp "<<sc<<" nodes "<<nodes<<" time "<<el
                <<" pv "<<uci::moveToUci(best)<<"\n";
            if (sc>MATE_TH || sc<-MATE_TH) break;
            if (tlim>0 && el>=tlim) break;
        }
        return best;
    }
private:
    chrono::steady_clock::time_point start;
    long long tlim=-1, nodes=0; bool stop=false;
    array<array<Move,2>,128> killers;
    array<array<int,64>,64> history;
    static const size_t TT_SIZE=1<<22, TT_MASK=TT_SIZE-1;
    vector<TTEntry> tt=vector<TTEntry>(TT_SIZE);

    long long ms(){ return chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now()-start).count(); }
    bool timeUp(){ if ((++nodes%2048)==0 && tlim>0 && ms()>=tlim) stop=true; return stop; }

    int pieceVal(PieceType t){
        if (t==PieceType::PAWN) return 100; if (t==PieceType::KNIGHT) return 320;
        if (t==PieceType::BISHOP) return 330; if (t==PieceType::ROOK) return 500;
        if (t==PieceType::QUEEN) return 900; return 0;
    }
    // ordering: TT move, then captures (MVV-LVA), killers, history
    void order(Board& b, Movelist& m, Move ttm, int ply){
        int n=m.size(); int sc[256];
        for (int i=0;i<n;i++){
            Move mv=m[i];
            if (mv==ttm) sc[i]=2000000;
            else if (b.isCapture(mv)) sc[i]=1000000+pieceVal(b.at(mv.to()).type())*100-pieceVal(b.at(mv.from()).type());
            else if (mv==killers[ply][0]) sc[i]=900000;
            else if (mv==killers[ply][1]) sc[i]=800000;
            else sc[i]=history[mv.from().index()][mv.to().index()];
        }
        for (int i=0;i<n;i++){ int bj=i; for (int j=i+1;j<n;j++) if (sc[j]>sc[bj]) bj=j;
            if (bj!=i){ swap(sc[i],sc[bj]); swap(m[i],m[bj]); } }
    }
    bool hasPieces(const Board& b, Color c){
        return (bool)b.pieces(PieceType::KNIGHT,c)||(bool)b.pieces(PieceType::BISHOP,c)
            ||(bool)b.pieces(PieceType::ROOK,c)||(bool)b.pieces(PieceType::QUEEN,c);
    }

    // quiescence: resolve captures so we don't evaluate mid-trade
    int quiesce(Board& b, int alpha, int beta, int ply){
        if (timeUp()) return 0;
        bool chk=b.inCheck();
        int stand=-INF;
        if (!chk){ stand=evalPos(b); if (stand>=beta) return stand; if (stand>alpha) alpha=stand; }
        Movelist m;
        if (chk) movegen::legalmoves(m,b);
        else movegen::legalmoves<movegen::MoveGenType::CAPTURE>(m,b);
        if (m.empty()) return chk?-MATE+ply:stand;
        order(b,m,Move::NO_MOVE,ply);
        int best=stand;
        for (const auto& mv:m){
            b.makeMove(mv); int s=-quiesce(b,-beta,-alpha,ply+1); b.unmakeMove(mv);
            if (stop) return best>-INF?best:0;
            if (s>best) best=s; if (s>alpha) alpha=s; if (alpha>=beta) break;
        }
        return best;
    }

    int searchRoot(Board& b, int depth, Move& best){
        Movelist m; movegen::legalmoves(m,b);
        order(b,m,best,0);
        int alpha=-INF, bestScore=-INF; bool first=true;
        for (const auto& mv:m){
            b.makeMove(mv);
            int s;
            if (first) s=-search(b,depth-1,-INF,-alpha,1,true);
            else { s=-search(b,depth-1,-alpha-1,-alpha,1,true);
                   if (s>alpha) s=-search(b,depth-1,-INF,-alpha,1,true); }
            b.unmakeMove(mv);
            if (stop) break;
            if (s>bestScore){ bestScore=s; best=mv; }
            if (s>alpha) alpha=s;
            first=false;
        }
        return bestScore;
    }

    int search(Board& b, int depth, int alpha, int beta, int ply, bool canNull){
        if (timeUp()) return 0;
        if (ply>0 && (b.isRepetition(1) || b.isHalfMoveDraw())) return 0;
        int alpha0=alpha; bool pv=(beta-alpha)>1;

        TTEntry& e=tt[b.hash()&TT_MASK]; Move ttm=Move::NO_MOVE;
        if (e.key==b.hash()){
            ttm=e.move;
            if (e.depth>=depth && !pv){
                if (e.flag==TT_EXACT) return e.score;
                if (e.flag==TT_LOWER && e.score>=beta) return e.score;
                if (e.flag==TT_UPPER && e.score<=alpha) return e.score;
            }
        }
        bool chk=b.inCheck();
        if (chk) depth++;                       // check extension
        if (depth<=0) return quiesce(b,alpha,beta,ply);

        // null-move pruning
        if (canNull && !pv && !chk && depth>=3 && hasPieces(b,b.sideToMove()) && beta<MATE_TH){
            int R=2+depth/6;
            b.makeNullMove();
            int s=-search(b,depth-1-R,-beta,-beta+1,ply+1,false);
            b.unmakeNullMove();
            if (stop) return 0;
            if (s>=beta) return beta;
        }

        Movelist m; movegen::legalmoves(m,b);
        if (m.empty()) return chk?-MATE+ply:0;
        order(b,m,ttm,ply);

        int best=-INF; Move bm=m[0]; int done=0;
        for (int i=0;i<m.size();i++){
            Move mv=m[i];
            bool cap=b.isCapture(mv);
            bool gc=b.givesCheck(mv)!=CheckType::NO_CHECK;
            b.makeMove(mv);
            int s;
            if (done==0) s=-search(b,depth-1,-beta,-alpha,ply+1,true);
            else {
                int red=(depth>=3 && done>=3 && !cap && !gc && !chk)?1:0;   // late move reduction
                s=-search(b,depth-1-red,-alpha-1,-alpha,ply+1,true);
                if (s>alpha && red) s=-search(b,depth-1,-alpha-1,-alpha,ply+1,true);
                if (s>alpha && s<beta) s=-search(b,depth-1,-beta,-alpha,ply+1,true);
            }
            b.unmakeMove(mv);
            if (stop) return best>-INF?best:0;
            done++;
            if (s>best){ best=s; bm=mv; }
            if (s>alpha) alpha=s;
            if (alpha>=beta){
                if (!cap){
                    if (!(mv==killers[ply][0])){ killers[ply][1]=killers[ply][0]; killers[ply][0]=mv; }
                    history[mv.from().index()][mv.to().index()]+=depth*depth;
                }
                break;
            }
        }
        unsigned char flag=(best<=alpha0)?TT_UPPER:(best>=beta)?TT_LOWER:TT_EXACT;
        if (depth>=e.depth || e.key!=b.hash()){ e.key=b.hash(); e.score=best; e.depth=(short)depth; e.flag=flag; e.move=bm; }
        return best;
    }
};

Board board; ChessEngine engine;

void parsePosition(const string& line){
    stringstream ss(line); string t; ss>>t; ss>>t;
    if (t=="startpos") board.setFen(constants::STARTPOS);
    else if (t=="fen"){ string fen; for (int i=0;i<6;i++){ ss>>t; fen+=t; if (i!=5) fen+=" "; } board.setFen(fen); }
    if (ss>>t && t=="moves") while (ss>>t) board.makeMove(uci::uciToMove(board,t));
}
int pDepth=-1; long long pMove=-1,pWt=-1,pBt=-1,pWi=0,pBi=0; int pMtg=-1;
void parseGo(const string& line){
    stringstream ss(line); string t; ss>>t;
    pDepth=-1; pMove=-1; pWt=-1; pBt=-1; pWi=0; pBi=0; pMtg=-1;
    while (ss>>t){
        if (t=="depth") ss>>pDepth; else if (t=="movetime") ss>>pMove;
        else if (t=="wtime") ss>>pWt; else if (t=="btime") ss>>pBt;
        else if (t=="winc") ss>>pWi; else if (t=="binc") ss>>pBi; else if (t=="movestogo") ss>>pMtg;
    }
}
int main(int argc, char** argv){
    if (argc>1 && g_net.load(argv[1])) cout<<"info string hybrid eval (classical + NN "<<argv[1]<<")\n";
    else cout<<"info string classical eval\n";
    board.setFen(constants::STARTPOS);
    string line; cout.setf(ios::unitbuf);
    while (getline(cin,line)){
        if (line=="uci") cout<<"id name RajitBot\nid author Rajit\nuciok\n";
        else if (line=="isready") cout<<"readyok\n";
        else if (line=="ucinewgame") board.setFen(constants::STARTPOS);
        else if (line.rfind("position",0)==0) parsePosition(line);
        else if (line.rfind("go",0)==0){
            parseGo(line);
            int ud=64; long long ut=-1;
            if (pDepth>0){ ud=pDepth; ut=-1; }
            else if (pMove>0) ut=pMove;
            else {
                bool white=board.sideToMove()==Color::WHITE;
                long long myt=white?pWt:pBt, myi=white?pWi:pBi;
                if (myt>0){ int mtg=pMtg>0?pMtg:30; ut=myt/mtg+myi/2-50; if (ut<50) ut=50; }
                else { ud=9; ut=-1; }
            }
            Move best=engine.bestMove(board,ud,ut);
            cout<<"bestmove "<<uci::moveToUci(best)<<endl;
        }
        else if (line=="quit") break;
    }
}
