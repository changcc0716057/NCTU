/*
Team name: eat_or_dai	
Group ID: 9
Members: 0716057 張家誠
		 0716209 戴靖婷
		 0716231 黃嘉渝
*/

#include "STcpClient.h"
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <memory>
#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <chrono>

/*
	輪到此程式移動棋子
	board : 棋盤狀態(vector of vector), board[l][i][j] = l layer, i row, j column 棋盤狀態(l, i, j 從 0 開始)
			0 = 空、1 = 黑、2 = 白、-1 = 四個角落
	is_black : True 表示本程式是黑子、False 表示為白子

	return Step
	Step : vector, Step = {r, c}
			r, c 表示要下棋子的座標位置 (row, column) (zero-base)
*/

/* Global Variables */
typedef std::vector<std::vector<std::vector<int>>> Board;
std::vector<std::vector<int>> get_available_move(Board &p_board);
unsigned get_random_idx(unsigned p_size);
unsigned cnt = 0;

/* Node for MCTS */
class Node
{
	public:
		typedef std::vector<Node*> Childs;

		/* Constructor and Destructor */
		Node(Node *p_ParentNode, Board &p_board, std::vector<int> &p_move, unsigned p_pieces, int p_score, unsigned p_lines, unsigned p_color)
			: ParenetNode(p_ParentNode), ChildNode(std::vector<Node*>()), board(p_board), unexpanded_move(get_available_move(p_board)),
			move(p_move), pieces(p_pieces), visit(0), win(0), score(p_score), lines(p_lines), color(p_color) {}
		~Node() = default;

		/* return attributes */
		Childs GetChilds() { return ChildNode; }
		std::vector<int> GetMove() { return move; }
		unsigned GetPieces() { return pieces; }
		unsigned GetColor() { return color; }
		unsigned GetLines() { return lines; }
		int GetScore() { return score; }
		std::vector<std::vector<int>> GetUnexpandedMove() { return unexpanded_move; }
		std::vector<int> GetLastMove(Board &p_board);
		Board GetBoard() { return board; }

		/* Checking function */
		bool is_root() { return (ParenetNode == NULL); }
		bool can_expand() { return (unexpanded_move.size() != 0); }

		/* Calculate the UCB of node */
		double UCB(double exploration_ratio) {
			if (visit == 0)
				return 0;
			double exploitation = (double)(win) / visit;
			double exploration = exploration_ratio * sqrt(2*log(ParenetNode->visit) / visit);
			return exploitation + exploration;
		}

		void addChild(Node* newchild) {
			ChildNode.push_back(newchild);
		}

		/* Delete Node */
		void deletenodes();

		/* Some MCTS functions */
		Node* select(double exploration_ratio);
		Node* expand(unsigned our_color);
		void backpropagate(bool win, unsigned our_color);
		bool rollout(unsigned our_color);

	private:
		Node *ParenetNode;
		Childs ChildNode;
		Board board;
		std::vector<std::vector<int>> unexpanded_move;
		std::vector<int> move; // how to get to this node from previous node
		unsigned pieces;
		unsigned visit;
		int win;
		int score;      // difference between my score and opponent's score
		unsigned lines;
		unsigned color; // 1:black, 2:white
};

class MCTS
{
	public:
		MCTS() : root(NULL), curNode(NULL), our_color(0) {}
		~MCTS() = default;

		Node* traverse_to_leaf();
		std::vector<int> GetAction(Board &board, bool is_black);
		void simulation();

	private:
		Node* root;
		Node* curNode;
		unsigned our_color;
};

// 用來得到要下哪個位置才能到這個 board
std::vector<int> Node::GetLastMove(Board &p_board)
{
	std::vector<int> step(3, -1);
	for (int l = 0; l < 6; l++) {
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 6; j++) {
				if(this->board[l][i][j] != p_board[l][i][j]) {
					step = {l, i, j};
					return step;
				}
			}
		}
	}
	return step;
}

/* Calculate the scores of the current move */
void calculate_move_score(std::vector<std::vector<std::vector<int>>> &board, std::vector<int> &move, unsigned our_color, unsigned cur_color, unsigned &lines, int &score)
{
	/* move[l][i][j] = lth layer, ith row, jth column */
	/* Surface */
	// check vertical lines
	int addlines = 0;
	int cnt = 1;
	for(int i = move[1] - 1; i > -1 && i > move[1] - 4; --i) {
		if(cur_color == board[move[0]][i][move[2]]) ++cnt;
		else break;
	}
	for(int i = move[1] + 1; i < 6 && i < move[1] + 4; ++i) {
		if(cur_color == board[move[0]][i][move[2]]) ++cnt;
		else break;
	}
	while(cnt > 3) {
		++addlines; 
		--cnt;
	}
	// check horizontal lines
	cnt = 1;
	for(int j = move[2] - 1; j > -1 && j > move[2] - 4; --j) {
		if(cur_color == board[move[0]][move[1]][j]) ++cnt;
		else break;
	}
	for(int j = move[2] + 1; j < 6 && j < move[2] + 4; ++j) {
		if(cur_color == board[move[0]][move[1]][j]) ++cnt;
		else break;
	}
	while(cnt > 3) {
		++addlines;
		--cnt;
	}
    // check oblique lines (m = 1)
	if((move[2] - move[1]) == 2) {
		if(cur_color == board[move[0]][0][2] && cur_color == board[move[0]][1][3] &&
			cur_color == board[move[0]][2][4] && cur_color == board[move[0]][3][5]) {
				++addlines;
		}
	}
	else if(move[2] == move[1]) {
		if(cur_color == board[move[0]][1][1] && cur_color == board[move[0]][2][2] &&
			cur_color == board[move[0]][3][3] && cur_color == board[move[0]][4][4]) {
				++addlines;
		}
	}
	else if((move[2] - move[1]) == -2) {
		if(cur_color == board[move[0]][2][0] && cur_color == board[move[0]][3][1] &&
			cur_color == board[move[0]][4][2] && cur_color == board[move[0]][5][3]) {
				++addlines;
		}
	}
	// check oblique lines (m = -1)
	else if((move[1] + move[2]) == 3) {
		if(cur_color == board[move[0]][0][3] && cur_color == board[move[0]][1][2] &&
			cur_color == board[move[0]][2][1] && cur_color == board[move[0]][3][0]) {
				++addlines;
		}
	}
	else if((move[1] + move[2]) == 5) {
		if(cur_color == board[move[0]][1][4] && cur_color == board[move[0]][2][3] &&
			cur_color == board[move[0]][3][2] && cur_color == board[move[0]][4][1]) {
				++addlines;
		}
	}
	else if((move[1] + move[2]) == 7) {
		if(cur_color == board[move[0]][2][5] && cur_color == board[move[0]][3][4] &&
			cur_color == board[move[0]][4][3] && cur_color == board[move[0]][5][2]) {
				++addlines;
		}
	}
	// check deep line
	if(move[0] > 2 && cur_color == board[move[0]][move[1]][move[2]] && cur_color == board[move[0]-1][move[1]][move[2]]
					&& cur_color == board[move[0]-2][move[1]][move[2]] && cur_color == board[move[0]-3][move[1]][move[2]]) {
				++addlines;
	}
	// check vertical deep lines
	cnt = 1;
	for(int l = move[0] - 1, i = move[1] - 1; l > -1 && i > -1 && l > move[0] - 4 && i > move[1] - 4 ; --l, --i) {
		if(cur_color == board[l][i][move[2]]) ++cnt;
		else break;
	}
	for(int l = move[0] + 1, i = move[1] + 1; l < 6 && i < 6 && l < move[0] + 4 && i < move[1] + 4; ++l, ++i) {
		if(cur_color == board[l][i][move[2]]) ++cnt;
		else break;
	}
	while(cnt > 3) {
		++addlines;
		--cnt;
	}
	cnt = 1;
	for(int l = move[0] - 1, i = move[1] + 1; l > -1 && i < 6 && l > move[0] - 4 && i < move[1] + 4; --l, ++i) {
		if(cur_color == board[l][i][move[2]]) ++cnt;
		else break;
	}
	for(int l = move[0] + 1, i = move[1] - 1; l < 6 && i > -1 && l < move[0] + 4 && i > move[1] - 4; ++l, --i) {
		if(cur_color == board[l][i][move[2]]) ++cnt;
		else break;
	}
	while(cnt > 3) {
		++addlines; 
		--cnt;
	}
	// check horizontal deep lines
	cnt = 1;
	for(int l = move[0] - 1, j = move[2] - 1; l > -1 && j > -1 && l > move[0] - 4 && j > move[2] - 4; --l, --j) {
		if(cur_color == board[l][move[1]][j]) ++cnt;
		else break;
	}
	for(int l = move[0] + 1, j = move[2] + 1; l < 6 && j < 6 && l < move[0] + 4 && j < move[2] + 4; ++l, ++j) {
		if(cur_color == board[l][move[1]][j]) ++cnt;
		else break;
	}
	while(cnt > 3) {
		++addlines; 
		--cnt;
	}
	cnt = 1;
	for(int l = move[0] - 1, j = move[2] + 1; l > -1 && j < 6 && l > move[0] - 4 && j < move[2] + 4; --l, ++j) {
		if(cur_color == board[l][move[1]][j]) ++cnt;
		else break;
	}
	for(int l = move[0] + 1, j = move[2] - 1; l < 6 && j > -1 && l < move[0] + 4 && j > move[2] - 4; ++l, --j) {
		if(cur_color == board[l][move[1]][j]) ++cnt;
		else break;
	}
	while(cnt > 3) {
		++addlines; 
		--cnt;
	}

	// check oblique deep lines (m = 1)
	if((move[2] - move[1]) == 2) {
		if((move[1] == 0 && move[0] > 2 && move[0] < 6) || (move[1] == 1 && move[0] > 1 && move[0] < 5) || 
			(move[1] == 2 && move[0] > 0 && move[0] < 4) || (move[1] == 3 && move[0] < 3)) {
			int height = move[0] + move[1];
			if(cur_color == board[height][0][2] && cur_color == board[height - 1][1][3] &&
			   cur_color == board[height - 2][2][4] && cur_color == board[height - 3][3][5]) {
				++addlines;
			}
		}
		if((move[1] == 0 && move[0] < 3) || (move[1] == 1 && move[0] > 0 && move[0] < 4) || 
			(move[1] == 2 && move[0] > 1 && move[0] < 5) || (move[1] == 3 && move[0] > 2 && move[0] < 6)) {
			int height = move[0] - move[1];
			if(cur_color == board[height][0][2] && cur_color == board[height + 1][1][3] &&
			   cur_color == board[height + 2][2][4] && cur_color == board[height + 3][3][5]) {
				++addlines;
			}
		}
	}
	else if(move[2] == move[1]) {
		if((move[1] == 1 && move[0] > 2 && move[0] < 6) || (move[1] == 2 && move[0] > 1 && move[0] < 5) || 
			(move[1] == 3 && move[0] > 0 && move[0] < 4) || (move[1] == 4 && move[0] < 3)) {
			int height = move[0] + move[1] - 1;
			if(cur_color == board[height][1][1] && cur_color == board[height - 1][2][2] &&
			   cur_color == board[height - 2][3][3] && cur_color == board[height - 3][4][4]) {
				++addlines;
			}
		}
		if((move[1] == 1 && move[0] < 3) || (move[1] == 2 && move[0] > 0 && move[0] < 4) || 
			(move[1] == 3 && move[0] > 1 && move[0] < 5) || (move[1] == 4 && move[0] > 2 && move[0] < 6)) {
			int height = move[0] - move[1] + 1;
			if(cur_color == board[height][1][1] && cur_color == board[height + 1][2][2] &&
			   cur_color == board[height + 2][3][3] && cur_color == board[height + 3][4][4]) {
				++addlines;
			}
		}
	}
	else if((move[2] - move[1]) == -2) {
		if((move[1] == 2 && move[0] > 2 && move[0] < 6) || (move[1] == 3 && move[0] > 1 && move[0] < 5) || 
			(move[1] == 4 && move[0] > 0 && move[0] < 4) || (move[1] == 5 && move[0] < 3)) {
			int height = move[0] + move[1] - 2;
			if(cur_color == board[height][2][0] && cur_color == board[height - 1][3][1] &&
			   cur_color == board[height - 2][4][2] && cur_color == board[height - 3][5][3]) {
				++addlines;
			}
		}
		if((move[1] == 2 && move[0] < 3) || (move[1] == 3 && move[0] > 0 && move[0] < 4) || 
			(move[1] == 4 && move[0] > 1 && move[0] < 5) || (move[1] == 5 && move[0] > 2 && move[0] < 6)) {
			int height = move[0] - move[1] + 2;
			if(cur_color == board[height][2][0] && cur_color == board[height + 1][3][1] &&
			   cur_color == board[height + 2][4][2] && cur_color == board[height + 3][5][3]) {
				++addlines;
			}
		}
	}
	// check oblique lines (m = -1)
	else if((move[1] + move[2]) == 3 ) {
		if((move[1] == 0 && move[0] > 2 && move[0] < 6) || (move[1] == 1 && move[0] > 1 && move[0] < 5) || 
			(move[1] == 2 && move[0] > 0 && move[0] < 4) || (move[1] == 3 && move[0] < 3)) {
			int height = move[0] - move[2];
			if(cur_color == board[height][3][0] && cur_color == board[height + 1][2][1] &&
			   cur_color == board[height + 2][1][2] && cur_color == board[height + 3][0][3]) {
				++addlines;
			}
		}
		if((move[1] == 0 && move[0] < 3) || (move[1] == 1 && move[0] > 0 && move[0] < 4) || 
			(move[1] == 2 && move[0] > 1 && move[0] < 5) || (move[1] == 3 && move[0] > 2 && move[0] < 6)) {
			int height = move[0] + move[2];
			if(cur_color == board[height][3][0] && cur_color == board[height - 1][2][1] &&
			   cur_color == board[height - 2][1][2] && cur_color == board[height - 3][0][3]) {
				++addlines;
			}
		}
	}
	else if((move[1] + move[2]) == 5) {
		if((move[1] == 1 && move[0] > 2 && move[0] < 6) || (move[1] == 2 && move[0] > 1 && move[0] < 5) || 
			(move[1] == 4 && move[0] > 0 && move[0] < 4) || (move[1] == 4 && move[0] < 3)) {
			int height = move[0] - move[2] + 1;
			if(cur_color == board[height][4][1] && cur_color == board[height + 1][3][2] &&
			   cur_color == board[height + 2][2][3] && cur_color == board[height + 3][1][4]) {
				++addlines;
			}
		}
		if((move[1] == 1 && move[0] < 3) || (move[1] == 2 && move[0] > 0 && move[0] < 4) || 
			(move[1] == 3 && move[0] > 1 && move[0] < 5) || (move[1] == 4 && move[0] > 2 && move[0] < 6)) {
			int height = move[0] + move[2] - 1;
			if(cur_color == board[height][4][1] && cur_color == board[height - 1][3][2] &&
			   cur_color == board[height - 2][2][3] && cur_color == board[height - 3][1][4]) {
				++addlines;
			}
		}
	}
	else if((move[1] + move[2]) == 7) {
		if((move[1] == 2 && move[0] > 2 && move[0] < 6) || (move[1] == 3 && move[0] > 1 && move[0] < 5) || 
			(move[1] == 4 && move[0] > 0 && move[0] < 4) || (move[1] == 5 && move[0] < 3)) {
			int height = move[0] - move[2] + 2;
			if(cur_color == board[height][5][2] && cur_color == board[height + 1][4][3] &&
			   cur_color == board[height + 2][3][4] && cur_color == board[height + 3][2][5]) {
				++addlines;
			}
		}
		if((move[1] == 2 && move[0] < 3) || (move[1] == 3 && move[0] > 0 && move[0] < 4) || 
			(move[1] == 4 && move[0] > 1 && move[0] < 5) || (move[1] == 5 && move[0] > 2 && move[0] < 6)) {
			int height = move[0] + move[2] - 2;
			if(cur_color == board[height][5][2] && cur_color == board[height - 1][4][3] &&
			   cur_color == board[height - 2][3][4] && cur_color == board[height - 3][2][5]) {
				++addlines;
			}
		}
	}
	for (int i = 0; i < addlines; i++) {
		++lines;
		if (our_color == cur_color)
			score += floor((double)(100) / lines);
		else
			score -= floor((double)(100) / lines);
	}
}

Node* Node::select(double exploration_ratio = sqrt(2))
{
	double max_ucb = -1000.0;
	int max_id = -1;
	for (int i = 0; i < ChildNode.size(); i++) {
		double cur_ucb = ChildNode[i]->UCB(exploration_ratio);
		if (cur_ucb > max_ucb) {
			max_ucb = cur_ucb;
			max_id = i;
		}
	}
	Node *ret = ChildNode[max_id];
	if (exploration_ratio == 0) {
		for (int i = ChildNode.size()-1; i > -1; i--) {
			// std::cout << ChildNode[i]->UCB(exploration_ratio) << ' ' << ChildNode[i]->GetScore() << '\n';
			Node *cur = ChildNode[i];
			ChildNode.pop_back();
			if (i != max_id) {
				cur->deletenodes();
				delete cur;		
			}
		}
		ChildNode.push_back(ret);
	}
	return ret;
}

Node* Node::expand(unsigned our_color)
{
	// 在還沒展開過的 nodes 中選擇一個
	std::vector<int> move = this->unexpanded_move.back();
	this->unexpanded_move.pop_back();

	// 建立下一個 Board
	Board cur_board(this->board);
	unsigned next_color = (this->color == 1) ? 2 : 1;
	unsigned next_lines = this->GetLines();
	int next_score = this->GetScore();
	cur_board[move[0]][move[1]][move[2]] = next_color;
	calculate_move_score(cur_board, move, our_color, next_color, next_lines, next_score);
	Node* next_node = new Node(this, cur_board, move, this->pieces+1, next_score, next_lines, next_color);
	this->ChildNode.push_back(next_node);

	return next_node;
}

void Node::backpropagate(bool win, unsigned our_color)
{
	this->visit++;
	if ((win && this->color == our_color) || (!win && this->color != our_color))
		this->win++;
	else 
		this->win--;
	if (!this->is_root())
		this->ParenetNode->backpropagate(win, our_color);
}

bool Node::rollout(unsigned our_color)
{
	Board cur_board(this->board);
	// 記錄當前狀態的 valid move
	std::vector<std::vector<int>> available_move = get_available_move(cur_board);

	unsigned cur_pieces = this->pieces;
	int cur_score = this->score;
	unsigned cur_line = this->lines;
	unsigned next_color = (this->color == 1) ? 2 : 1;
	while(cur_pieces < 64) {
		// 得到隨機的一步
		unsigned random_idx = get_random_idx(available_move.size());
		std::vector<int> move = available_move[random_idx];
		cur_board[move[0]][move[1]][move[2]] = next_color;

		// 檢查這個位置是否能再放棋子
		available_move[random_idx][0]++;
		if (available_move[random_idx][0] > 5)
			available_move.erase(available_move.begin() + random_idx);
		calculate_move_score(cur_board, move, our_color, next_color, cur_line, cur_score);
		next_color = (next_color == 1) ? 2 : 1;
		cur_pieces++;
	}
	return (cur_score > 20) ? true : false;
}

void Node::deletenodes()
{
	for (auto childnode: this->ChildNode) {
		childnode->deletenodes();
		delete childnode;
	}
}

std::vector<std::vector<int>> get_available_move(Board &p_board)
{
	std::vector<std::vector<int>> available_move;
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			if (p_board[0][i][j] == -1)
				continue;
			for (int l = 0; l < 6; l++) {
				if (p_board[l][i][j] == 0) {
					std::vector<int> move = {l, i, j};
					available_move.push_back(move);
					break;
				}
			}
		}
	}
	return available_move;
}

unsigned get_random_idx(unsigned p_size)
{
	// auto current = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
	// unsigned d = current.count();
	std::srand(cnt);
	cnt++;
	unsigned long long rnd = std::rand();
	return (rnd)%p_size;
}

Node* MCTS::traverse_to_leaf()
{
	Node* cur_Node = this->curNode;
	while(true) {
		if (cur_Node->GetPieces() == 64)
			break;
		if (cur_Node->can_expand()) {
			cur_Node = cur_Node->expand(our_color);
			break;
		}
		else
			cur_Node = cur_Node->select();
	}
	return cur_Node;
}

void MCTS::simulation()
{
	Node* leaf = this->traverse_to_leaf();
	bool win = leaf->rollout(this->our_color);
	leaf->backpropagate(win, our_color);
}

std::vector<int> MCTS::GetAction(Board &board, bool is_black)
{
	// Initialization
	time_t start_time = time(NULL);
	if (this->root == NULL) {
		unsigned p_color = is_black ? 2 : 1;
		unsigned p_piece = is_black ? 0 : 1;
		std::vector<int> p_move = {-1, -1, -1};
		this->root = new Node(NULL, board, p_move, p_piece, 0, 0, p_color);
		this->curNode = this->root;
		this->our_color = is_black ? 1 : 2;
	}
	// New turn
	else {
		bool find = false;
		std::vector<int> opponent_move = curNode->GetLastMove(board);
		for (auto &child : curNode->GetChilds()) {
			if (child->GetMove() == opponent_move) {
				curNode = child;
				find = true;
				break;
			}
		}
		if (!find) {
			for (auto &unexpands : curNode->GetUnexpandedMove()) {
				if (unexpands == opponent_move) {
					unsigned cur_lines = curNode->GetLines();
					int cur_score = curNode->GetScore();
					calculate_move_score(board, opponent_move, this->our_color, root->GetColor(), cur_lines, cur_score);
					Node* next_node = new Node(curNode, board, opponent_move, curNode->GetPieces()+1, cur_score, cur_lines, root->GetColor());
					curNode->addChild(next_node);
					curNode = next_node;
					break;
				}
			}
		}
	}
	
	while(time(NULL) - start_time <= 4) {
		this->simulation();
	}
	
	curNode = curNode->select(0);
	std::vector<int> ret_move = {(curNode->GetMove())[1], (curNode->GetMove())[2]};

	if ((is_black && (curNode->GetPieces() == 63)) || ((!is_black) && (curNode->GetPieces() == 64))) {
		root->deletenodes();
		delete root;
		root = NULL;
		curNode = NULL;
	}
	return ret_move;
}

MCTS mcts;

std::vector<int> GetStep(std::vector<std::vector<std::vector<int>>> &board, bool is_black)
{
	std::vector<int> step = mcts.GetAction(board, is_black);
	return step;
}

int main()
{
	int id_package;
	std::vector<std::vector<std::vector<int>>> board;
	std::vector<int> step;

	bool is_black;
	while (true)
	{
		if (GetBoard(id_package, board, is_black))
			break;

		step = GetStep(board, is_black);
		SendStep(id_package, step);
	}
}