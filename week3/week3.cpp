// USAGE: g++ -std=c++17 -O3 week3.cpp -o engine


#include <fstream>
#include <iostream>
// #include <nlohmann/json.hpp>
#include "chess.hpp"
#include <cmath>

// using json = nlohmann::json;
using namespace chess;
using namespace std;

class ChessEngine{
public:
    Move bestMove(Board& board, float depth) {
        Movelist moves;
        movegen::legalmoves(moves,board);
        
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
    float alphaBeta(Board& board, float depth, float alpha, float beta, bool maximizer){
        Movelist moves;
        movegen::legalmoves(moves,board);    
        
        if (moves.size() ==0){
            if (board.inCheck()){
                if (maximizer){
                    return -1;
                }
                else{
                    return 1;
                }
            }
            else{
                return 0;
            }
        }
        else if (depth==0){
            return 0;
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

int main(){
    string fen;
    getline(cin, fen);
    int matein;
    cin >> matein;
    // string fen = "4kb1r/p2n1ppp/4q3/4p1B1/4P3/1Q6/PPP2PPP/2KR4 w k - 1 0";

    Board board;
    board.setFen(fen);

    ChessEngine engine;
    Move best=engine.bestMove(board, 2*matein);
    cout << "Best move: " << uci::moveToSan(board, best) << endl;
}

// int main() {
//     std::ifstream file("mate_in_2.json");

//     json puzzles;
//     file >> puzzles;

//     ChessEngine engine;

//     int total = 0;
//     int correct = 0;

//     for (auto& [fen, solution] : puzzles.items()) {
//         Board board;
//         board.setFen(fen);

//         Move best = engine.bestMove(board, 4);

//         std::string engineMove = uci::moveToSan(board, best);

//         std::string solutionLine = solution.get<std::string>();

//         auto isMoveNumber = [](const std::string& s) {
//             return !s.empty() &&
//                 s.back() == '.' &&
//                 std::all_of(s.begin(), s.end() - 1, ::isdigit);
//         };

//         std::stringstream ss(solutionLine);
//         std::string token;
//         std::string expectedFirstMove;

//         while (ss >> token) {
//             if (isMoveNumber(token))
//                 continue;

//             expectedFirstMove = token;
//             break;
//         }

//         total++;

//         if (engineMove == expectedFirstMove) {
//             correct++;
//         }
//     }

//     std::cout << "Score: " << correct << "/" << total << '\n';
// }