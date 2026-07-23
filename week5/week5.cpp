//USAGE: g++ -std=c++17 -O3 -static week4.cpp -o engine.exe

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

    float evaluate(Board& board) {
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

int main() {
    cout << "STARTED" << "\n";
    board.setFen(constants::STARTPOS);

    string line;
    cout.setf(ios::unitbuf);
    while (getline(cin, line)) {
        if (line == "uci") {
            cout << "id name RajitBot\nid author Rajit\nuciok\n";
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