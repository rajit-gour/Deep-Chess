
#include <fstream>
#include <iostream>
#include <sstream>
#include "chess.hpp"
#include <cmath>

using namespace chess;
using namespace std;

class ChessEngine{
public:
    Move bestMove(Board& board, float depth) {
        Movelist moves;
        movegen::legalmoves(moves,board);

        if (moves.empty()) {
            return Move::NO_MOVE;
        }
        Move best=moves[0];
        float score=-INFINITY;

        for (const auto& move : moves){
            board.makeMove(move);
            float newscore=alphaBeta(board,depth -1, -INFINITY,INFINITY,false);
            board.unmakeMove(move);

            if (newscore>score){
                score=newscore;
                best=move;
            }
        }
        return best;
    }
private:
    float evaluate(Board& board) {
        float score = 0;
        Movelist mobilityMoves;
        movegen::legalmoves(mobilityMoves, board);
        if (board.sideToMove() == Color::WHITE)
            score += 0.1 * mobilityMoves.size();
        else
            score -= 0.1 * mobilityMoves.size();
        for (int sq = 0; sq < 64; sq++) {
            Piece p = board.at(Square(sq));
            if (p == Piece::NONE) continue;
            
            float val = 0;
            PieceType type = p.type();

            if      (type == PieceType::PAWN)   val = 1;
            else if (type == PieceType::KNIGHT) val = 3;
            else if (type == PieceType::BISHOP) val = 3;
            else if (type == PieceType::ROOK)   val = 5;
            else if (type == PieceType::QUEEN)  val = 9;
            else                                 val = 0;
            
            if (p.color() == Color::WHITE) score += val;
            else score -= val;
        }
        return score;
    }
    float alphaBeta(Board& board, float depth, float alpha, float beta, bool maximizer){
        Movelist moves;
        movegen::legalmoves(moves,board);    
        
        if (moves.size() ==0){
            if (board.inCheck()){
                if (maximizer){
                    return -10000-depth;
                }
                else{
                    return 10000+depth;
                }
            }
            else{
                return 0;
            }
        }
        else if (depth==0){
            return evaluate(board);
        }
        else if (maximizer){
            float maxi=-INFINITY;
            for (const auto& move : moves){
                board.makeMove(move);
                float newscore=alphaBeta(board,depth -1, alpha,beta,false);
                board.unmakeMove(move);
                maxi=max(maxi,newscore);
                alpha=max(alpha,newscore);
                if (beta<=alpha){
                    break;
                }
            }
            return maxi;
        }
        else{
            float mini=INFINITY;
            for (const auto& move : moves){
                board.makeMove(move);
                float newscore=alphaBeta(board,depth -1,alpha,beta,true);
                board.unmakeMove(move);
                mini=min(mini,newscore);
                beta=min(beta,newscore);
                if (beta<=alpha){
                    break;
                }
            }
            return mini;
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
            Move best = engine.bestMove(board,5); 
            cout << "bestmove " << uci::moveToUci(best) << endl;
        }
        else if (line == "quit") {
            break;
        }
    }
}
