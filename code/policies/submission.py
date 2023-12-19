import numpy as np
import random

class HeuristicGomokuAI:
    def __init__(self, board_size, win_size, lookahead_depth):
        self.board_size = board_size
        self.win_size = win_size
        self.lookahead_depth = lookahead_depth


    def evaluate_board(self, board, player_index):
        score = 0
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[player_index, x, y] == 1:
                    score += self.evaluate_position(board, x, y, player_index)
        return score

    def evaluate_position(self, board, x, y, player_index):
        position_score = 0
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        for dx, dy in directions:
            line_length, open_ends, blocked_ends, four_in_a_row, three_in_a_row = self.check_line(board, x, y, dx, dy, player_index)
            # existing scoring logic...
            position_score += four_in_a_row * 500  # new scoring for four-in-a-row
            position_score += three_in_a_row * 300  # new scoring for three-in-a-row
        return position_score

    def check_line(self, board, x, y, dx, dy, player_index):
        line_length = 0
        open_ends = 0
        blocked_ends = 0
        four_in_a_row = 0
        three_in_a_row = 0

        # Check line in one direction
        for i in range(self.win_size):
            nx, ny = x + dx * i, y + dy * i
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if board[player_index, nx, ny] == 1:
                    line_length += 1
                    if line_length == 4 and open_ends > 0:
                        four_in_a_row += 1
                    elif line_length == 3 and open_ends == 2:
                        three_in_a_row += 1
                elif board[0, nx, ny] == 1:
                    open_ends += 1
                    break
                else:
                    break

        # Check line in the opposite direction
        # ...

        return line_length, open_ends, blocked_ends, four_in_a_row, three_in_a_row

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or self.game_over(board):
            return self.evaluate_board(board, maximizing_player)

        if maximizing_player:
            max_eval = float('-inf')
            for child in self.get_children(board, maximizing_player):
                eval = self.minimax(child, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for child in self.get_children(board, not maximizing_player):
                eval = self.minimax(child, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    
    def make_move(self, board):
        best_score = float('-inf')
        best_move = None
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[0, x, y] == 0:  # if the cell is empty
                    board[1, x, y] = 1  # make a move
                    score = self.minimax(board, self.lookahead_depth, float('-inf'), float('inf'), False)
                    board[1, x, y] = 0  # undo the move
                    if score > best_score:
                        best_score = score
                        best_move = (x, y)
        return best_move

    def generate_moves(self, board, valid_actions):
        num_pieces = np.sum(board[1,:,:] + board[2,:,:])
        
        # First move at the center
        if num_pieces == 0:
            return [(self.board_size // 2, self.board_size // 2)]

        # Second move adjacent to the center, if those positions are empty
        if num_pieces == 2:
            return self.second_move_near_center(board, valid_actions)

        # After the second move, prioritize moves near the AI's existing pieces
        return self.get_moves_near_own_pieces(board, valid_actions)

    def second_move_near_center(self, board, valid_actions):
        center_x, center_y = self.board_size // 2, self.board_size // 2
        adjacent_positions = [(center_x - 1, center_y - 1), (center_x + 1, center_y - 1),
                              (center_x - 1, center_y + 1), (center_x + 1, center_y + 1)]
        valid_moves = [pos for pos in adjacent_positions if pos in valid_actions]
        return valid_moves if valid_moves else list(valid_actions)  # Fallback to any valid move

    def get_moves_near_own_pieces(self, board, valid_actions):
        player_index = 2 if 'x' == 'x' else 1  # Assuming 'x' represents the AI's pieces
        moves = set()
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[player_index, x, y] == 1:  # Check for AI's own pieces
                    for dx in range(-2, 3):  # Adjust the range for proximity
                        for dy in range(-2, 3):
                            nx, ny = x + dx, y + dy
                            if (nx, ny) in valid_actions:
                                moves.add((nx, ny))
        return list(moves) if moves else list(valid_actions)  # Fallback to any valid move

    def select_best_move(self, board, player, possible_moves):
        player_index = 2 if player == 'x' else 1
        opponent_index = 3 - player_index
        best_move = None
        best_score = -float('inf')

        # Check for opponent threats
        opponent_threat = self.is_threatening(board, opponent_index, 3)  # Check for line of 3

        for move in possible_moves:
            new_board = np.copy(board)
            new_board[0, move[0], move[1]] = 0
            new_board[player_index, move[0], move[1]] = 1
            score = self.evaluate_board(new_board, player_index)

            # Adjust strategy based on opponent threat
            if opponent_threat:
                score -= self.evaluate_board(new_board, opponent_index)

            if score > best_score:
                best_move = move
                best_score = score

        return best_move
        
    def is_threatening(self, board, player_index, line_length):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # All possible line directions
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[player_index, x, y] == 1:
                    for dx, dy in directions:
                        if self.check_line_threat(board, x, y, dx, dy, player_index, line_length):
                            return True
        return False

    def check_line_threat(self, board, x, y, dx, dy, player_index, line_length):
        count = 0
        for i in range(1, line_length):
            nx, ny = x + dx * i, y + dy * i
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if board[player_index, nx, ny] == 1:
                    count += 1
                elif board[0, nx, ny] == 1:  # Empty cell found
                    break
                else:  # Opponent's piece or edge of board
                    return False
        return count == line_length - 1
    
    def get_children(self, board, player_index):
        children = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[0, x, y] == 0:  # if the cell is empty
                    new_board = board.copy()
                    new_board[player_index, x, y] = 1  # make a move
                    children.append(new_board)
        return children
    
    def game_over(self, board):
        # Check rows, columns and diagonals for a winning line
        for x in range(self.board_size):
            for y in range(self.board_size):
                if any(self.check_line(board, x, y, dx, dy, player_index) >= self.win_size for dx, dy in [(1,0), (0,1), (1,1), (1,-1)] for player_index in [1, 2]):
                    return True
        return False

class Submission:
    def __init__(self, board_size, win_size):
        self.heuristic_ai = HeuristicGomokuAI(board_size, win_size, lookahead_depth=2)

    def __call__(self, state):
        current_player = state.current_player()  # Determine the current player from the state
        valid_moves = state.valid_actions()  # Get valid moves
        return self.heuristic_ai.select_best_move(state.board, current_player, self.heuristic_ai.generate_moves(state.board, valid_moves))


# Assuming that the `state` object has a `current_player()` method and a `board` attribute

# Example usage
if __name__ == "__main__":
    board_size = 15
    win_size = 5
    ai = HeuristicGomokuAI(board_size, win_size)
    board = np.zeros((board_size, board_size), dtype=int)  # 0 for empty, 1 for player 1, 2 for player 2
    current_player = 1

    while True:  # Add your own game-over condition check
        move = ai(board, current_player)
        board[move] = current_player
        current_player = 3 - current_player  # Switch player
        # Add display or logging of the board state
