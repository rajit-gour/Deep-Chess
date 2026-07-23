//USAGE: g++ -std=c++17 -O3 -static engine_nnue.cpp -o engine_nnue.exe
// (correctness build: add -DVERIFY_ACC to assert incremental==from-scratch)
// NNUE eval: two accumulators (White view, Black view) updated incrementally
// in make/unmake, so a moved piece is added/removed instead of recomputing.

#include <fstream>
#include <iostream>
#include <sstream>
#include "chess.hpp"
#include <cmath>
#include <chrono>
#include <algorithm>
#include <vector>
#include <array>
#include <cassert>

using namespace chess;
using namespace std;

// ---- NNUE weights ----
// FT: 768 -> ACC (shared by both perspectives). head: [own,opp] -> 32 -> 1.
struct Net {
    static const int IN = 768, MAXACC = 256, H1 = 32;
    int acc = 256;
    vector<float> ftW, ftB, l1W, l1B, l2W, l2B;
    bool loaded = false;

    bool load(const string& p){
        ifstream f(p); string tag; int in;
        f >> tag >> in >> acc;
        if (tag != "NNUE" || in != IN || acc > MAXACC) return false;
        ftW.resize((size_t)IN*acc); ftB.resize(acc);
        l1W.resize((size_t)H1*2*acc); l1B.resize(H1); l2W.resize(H1); l2B.resize(1);
        for (auto& v:ftW) f>>v; for (auto& v:ftB) f>>v;
        for (auto& v:l1W) f>>v; for (auto& v:l1B) f>>v;
        for (auto& v:l2W) f>>v; for (auto& v:l2B) f>>v;
        loaded = (bool)f; return loaded;
    }
    static int pieceIndex(PieceType t){
        if (t==PieceType::PAWN) return 0; if (t==PieceType::KNIGHT) return 1;
        if (t==PieceType::BISHOP) return 2; if (t==PieceType::ROOK) return 3;
        if (t==PieceType::QUEEN) return 4; return 5;
    }
    // feature index of a piece per perspective (own planes 0-5, enemy 6-11)
    int fW(Color c,int t,int sq) const { return ((c==Color::WHITE)?t:t+6)*64 + sq; }
    int fB(Color c,int t,int sq) const { return ((c==Color::BLACK)?t:t+6)*64 + (sq^56); }
};
Net g_net;

struct Acc { array<float,Net::MAXACC> w, b; };

static const int MATE=1000000, MATE_TH=MATE-1000, INF=MATE*2;
enum { TT_NONE, TT_EXACT, TT_LOWER, TT_UPPER };
struct TTEntry { uint64_t key=0; int score=0; short depth=-1; unsigned char flag=TT_NONE; Move move=Move::NO_MOVE; };

class ChessEngine {
public:
    Move bestMove(Board& board, int maxdepth, long long timelimit){
        start=chrono::steady_clock::now(); tlim=timelimit; stop=false; nodes=0;
        for (auto& k:killers){ k[0]=Move::NO_MOVE; k[1]=Move::NO_MOVE; }
        for (auto& r:history) r.fill(0);
        accStack.clear(); refresh(board);

        Movelist moves; movegen::legalmoves(moves,board);
        if (moves.empty()) return Move::NO_MOVE;
        Move best=moves[0];
        for (int d=1; d<=maxdepth; d++){
            Move bm=best; int sc=searchRoot(board,d,bm);
            if (stop) break;
            best=bm; long long el=ms();
            cout<<"info depth "<<d<<" score cp "<<sc<<" nodes "<<nodes<<" time "<<el
                <<" pv "<<uci::moveToUci(best)<<"\n";
            if (sc>MATE_TH||sc<-MATE_TH) break;
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
    Acc acc; vector<Acc> accStack;

    long long ms(){ return chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now()-start).count(); }
    bool timeUp(){ if ((++nodes%2048)==0 && tlim>0 && ms()>=tlim) stop=true; return stop; }

    // ---- accumulator ----
    void refresh(const Board& b){
        int n=g_net.acc;
        for (int j=0;j<n;j++){ acc.w[j]=g_net.ftB[j]; acc.b[j]=g_net.ftB[j]; }
        for (int sq=0;sq<64;sq++){ Piece p=b.at(Square(sq)); if (p!=Piece::NONE) add(p.color(),Net::pieceIndex(p.type()),sq,+1.f); }
    }
    inline void add(Color c,int t,int sq,float s){
        int n=g_net.acc;
        const float* rw=&g_net.ftW[(size_t)g_net.fW(c,t,sq)*n];
        const float* rb=&g_net.ftW[(size_t)g_net.fB(c,t,sq)*n];
        for (int j=0;j<n;j++){ acc.w[j]+=s*rw[j]; acc.b[j]+=s*rb[j]; }
    }
    // update the accumulator for a move (board is state BEFORE the move)
    void delta(const Board& b, Move mv){
        Color us=b.sideToMove(), them=~us;
        int from=mv.from().index(), to=mv.to().index(); auto ty=mv.typeOf();
        if (ty==Move::CASTLING){
            bool ks=(to%8)>(from%8);
            int kT,rT;
            if (us==Color::WHITE){ kT=ks?6:2; rT=ks?5:3; } else { kT=ks?62:58; rT=ks?61:59; }
            add(us,5,from,-1.f); add(us,3,to,-1.f); add(us,5,kT,+1.f); add(us,3,rT,+1.f);
            return;
        }
        int mt=Net::pieceIndex(b.at(Square(from)).type());
        add(us,mt,from,-1.f);
        if (ty==Move::ENPASSANT) add(them,0,(us==Color::WHITE)?to-8:to+8,-1.f);
        else { Piece cap=b.at(Square(to)); if (cap!=Piece::NONE) add(them,Net::pieceIndex(cap.type()),to,-1.f); }
        if (ty==Move::PROMOTION) add(us,Net::pieceIndex(mv.promotionType()),to,+1.f);
        else add(us,mt,to,+1.f);
    }
    inline void doMove(Board& b, Move mv){ accStack.push_back(acc); delta(b,mv); b.makeMove(mv);
#ifdef VERIFY_ACC
        { Acc s=acc; refresh(b); for (int j=0;j<g_net.acc;j++) assert(fabs(s.w[j]-acc.w[j])<1e-2 && fabs(s.b[j]-acc.b[j])<1e-2); acc=s; }
#endif
    }
    inline void undoMove(Board& b, Move mv){ b.unmakeMove(mv); acc=accStack.back(); accStack.pop_back(); }

    float relu(float x){ return x<0?0:x; }
    int evalPos(const Board& b){
        int n=g_net.acc;
        bool wtm=(b.sideToMove()==Color::WHITE);
        const auto& own=wtm?acc.w:acc.b; const auto& opp=wtm?acc.b:acc.w;
        float h[2*Net::MAXACC];
        for (int j=0;j<n;j++){ h[j]=relu(own[j]); h[n+j]=relu(opp[j]); }
        float o[Net::H1];
        for (int k=0;k<Net::H1;k++){ float s=g_net.l1B[k]; const float* row=&g_net.l1W[(size_t)k*2*n];
            for (int i=0;i<2*n;i++) s+=h[i]*row[i]; o[k]=relu(s); }
        float out=g_net.l2B[0]; for (int k=0;k<Net::H1;k++) out+=o[k]*g_net.l2W[k];
        float p=1.f/(1.f+expf(-out)); const float e=1e-6f; if (p<e)p=e; if (p>1-e)p=1-e;
        return (int)lround(400.0f*logf(p/(1-p)));
    }

    int pieceVal(PieceType t){
        if (t==PieceType::PAWN) return 100; if (t==PieceType::KNIGHT) return 320;
        if (t==PieceType::BISHOP) return 330; if (t==PieceType::ROOK) return 500;
        if (t==PieceType::QUEEN) return 900; return 0;
    }
    void order(Board& b, Movelist& m, Move ttm, int ply){
        int n=m.size(); int sc[256];
        for (int i=0;i<n;i++){ Move mv=m[i];
            if (mv==ttm) sc[i]=2000000;
            else if (b.isCapture(mv)) sc[i]=1000000+pieceVal(b.at(mv.to()).type())*100-pieceVal(b.at(mv.from()).type());
            else if (mv==killers[ply][0]) sc[i]=900000; else if (mv==killers[ply][1]) sc[i]=800000;
            else sc[i]=history[mv.from().index()][mv.to().index()]; }
        for (int i=0;i<n;i++){ int bj=i; for (int j=i+1;j<n;j++) if (sc[j]>sc[bj]) bj=j;
            if (bj!=i){ swap(sc[i],sc[bj]); swap(m[i],m[bj]); } }
    }
    bool hasPieces(const Board& b, Color c){
        return (bool)b.pieces(PieceType::KNIGHT,c)||(bool)b.pieces(PieceType::BISHOP,c)
            ||(bool)b.pieces(PieceType::ROOK,c)||(bool)b.pieces(PieceType::QUEEN,c);
    }

    int quiesce(Board& b, int alpha, int beta, int ply){
        if (timeUp()) return 0;
        bool chk=b.inCheck(); int stand=-INF;
        if (!chk){ stand=evalPos(b); if (stand>=beta) return stand; if (stand>alpha) alpha=stand; }
        Movelist m;
        if (chk) movegen::legalmoves(m,b); else movegen::legalmoves<movegen::MoveGenType::CAPTURE>(m,b);
        if (m.empty()) return chk?-MATE+ply:stand;
        order(b,m,Move::NO_MOVE,ply);
        int best=stand;
        for (const auto& mv:m){ doMove(b,mv); int s=-quiesce(b,-beta,-alpha,ply+1); undoMove(b,mv);
            if (stop) return best>-INF?best:0;
            if (s>best) best=s; if (s>alpha) alpha=s; if (alpha>=beta) break; }
        return best;
    }
    int searchRoot(Board& b, int depth, Move& best){
        Movelist m; movegen::legalmoves(m,b); order(b,m,best,0);
        int alpha=-INF, bestScore=-INF; bool first=true;
        for (const auto& mv:m){ doMove(b,mv);
            int s;
            if (first) s=-search(b,depth-1,-INF,-alpha,1,true);
            else { s=-search(b,depth-1,-alpha-1,-alpha,1,true); if (s>alpha) s=-search(b,depth-1,-INF,-alpha,1,true); }
            undoMove(b,mv);
            if (stop) break;
            if (s>bestScore){ bestScore=s; best=mv; } if (s>alpha) alpha=s; first=false; }
        return bestScore;
    }
    int search(Board& b, int depth, int alpha, int beta, int ply, bool canNull){
        if (timeUp()) return 0;
        if (ply>0 && (b.isRepetition(1)||b.isHalfMoveDraw())) return 0;
        int alpha0=alpha; bool pv=(beta-alpha)>1;
        TTEntry& e=tt[b.hash()&TT_MASK]; Move ttm=Move::NO_MOVE;
        if (e.key==b.hash()){ ttm=e.move;
            if (e.depth>=depth && !pv){
                if (e.flag==TT_EXACT) return e.score;
                if (e.flag==TT_LOWER && e.score>=beta) return e.score;
                if (e.flag==TT_UPPER && e.score<=alpha) return e.score; } }
        bool chk=b.inCheck(); if (chk) depth++;
        if (depth<=0) return quiesce(b,alpha,beta,ply);
        if (canNull && !pv && !chk && depth>=3 && hasPieces(b,b.sideToMove()) && beta<MATE_TH){
            int R=2+depth/6; b.makeNullMove();
            int s=-search(b,depth-1-R,-beta,-beta+1,ply+1,false);
            b.unmakeNullMove();
            if (stop) return 0; if (s>=beta) return beta; }
        Movelist m; movegen::legalmoves(m,b);
        if (m.empty()) return chk?-MATE+ply:0;
        order(b,m,ttm,ply);
        int best=-INF; Move bm=m[0]; int done=0;
        for (int i=0;i<m.size();i++){ Move mv=m[i];
            bool cap=b.isCapture(mv); bool gc=b.givesCheck(mv)!=CheckType::NO_CHECK;
            doMove(b,mv);
            int s;
            if (done==0) s=-search(b,depth-1,-beta,-alpha,ply+1,true);
            else { int red=(depth>=3 && done>=3 && !cap && !gc && !chk)?1:0;
                s=-search(b,depth-1-red,-alpha-1,-alpha,ply+1,true);
                if (s>alpha && red) s=-search(b,depth-1,-alpha-1,-alpha,ply+1,true);
                if (s>alpha && s<beta) s=-search(b,depth-1,-beta,-alpha,ply+1,true); }
            undoMove(b,mv);
            if (stop) return best>-INF?best:0;
            done++;
            if (s>best){ best=s; bm=mv; } if (s>alpha) alpha=s;
            if (alpha>=beta){ if (!cap){ if (!(mv==killers[ply][0])){ killers[ply][1]=killers[ply][0]; killers[ply][0]=mv; }
                history[mv.from().index()][mv.to().index()]+=depth*depth; } break; } }
        unsigned char flag=(best<=alpha0)?TT_UPPER:(best>=beta)?TT_LOWER:TT_EXACT;
        if (depth>=e.depth||e.key!=b.hash()){ e.key=b.hash(); e.score=best; e.depth=(short)depth; e.flag=flag; e.move=bm; }
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
    while (ss>>t){ if (t=="depth") ss>>pDepth; else if (t=="movetime") ss>>pMove;
        else if (t=="wtime") ss>>pWt; else if (t=="btime") ss>>pBt;
        else if (t=="winc") ss>>pWi; else if (t=="binc") ss>>pBi; else if (t=="movestogo") ss>>pMtg; }
}
int main(int argc, char** argv){
    string wp=(argc>1)?argv[1]:"nnue.txt";
    if (!g_net.load(wp)){ cout<<"info string failed to load "<<wp<<"\n"; return 1; }
    cout<<"info string NNUE acc="<<g_net.acc<<" ("<<wp<<")\n";
    board.setFen(constants::STARTPOS);
    string line; cout.setf(ios::unitbuf);
    while (getline(cin,line)){
        if (line=="uci") cout<<"id name RajitBot-NNUE\nid author Rajit\nuciok\n";
        else if (line=="isready") cout<<"readyok\n";
        else if (line=="ucinewgame") board.setFen(constants::STARTPOS);
        else if (line.rfind("position",0)==0) parsePosition(line);
        else if (line.rfind("go",0)==0){
            parseGo(line);
            int ud=64; long long ut=-1;
            if (pDepth>0){ ud=pDepth; ut=-1; }
            else if (pMove>0) ut=pMove;
            else { bool white=board.sideToMove()==Color::WHITE;
                long long myt=white?pWt:pBt, myi=white?pWi:pBi;
                if (myt>0){ int mtg=pMtg>0?pMtg:30; ut=myt/mtg+myi/2-50; if (ut<50) ut=50; } else { ud=9; ut=-1; } }
            Move best=engine.bestMove(board,ud,ut);
            cout<<"bestmove "<<uci::moveToUci(best)<<endl;
        }
        else if (line=="quit") break;
    }
}
