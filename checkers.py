import numpy as np
import pygame
import sys
import random
import time


pygame.init()


WIDTH, HEIGHT = 600, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
CELL_SIZE = WIDTH // 8
KING_RED = (200, 0, 0)
KING_BLUE = (0, 0, 200)

class Checkers:
    def __init__(self, use_alpha_beta=True, depth=3):
        self.board = self.create_board()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Checkers")
        self.selected_piece = None
        self.current_player = 'R'  # Red startsplayer controls Red
        self.use_alpha_beta = use_alpha_beta
        self.depth = depth
    #performance measures
        self.nodes_expanded = 0
        self.pruning_count = 0
        self.last_move_time = 0
        self.total_ai_time = 0
        self.move_count = 0

        self.draw_board()
    
    def create_board(self):
        board = np.full((8, 8), '-')
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:
                    board[row][col] = 'B'
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    board[row][col] = 'R'
        return board
    
    def draw_board(self):
        self.window.fill(WHITE)
        for row in range(8):
            for col in range(8):
                color = BLACK if (row + col) % 2 == 0 else WHITE
                pygame.draw.rect(self.window, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                piece = self.board[row][col]
                if piece in ['R', 'B']:
                    pygame.draw.circle(self.window, RED if piece == 'R' else BLUE,
                                       (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2 - 5)
                elif piece in ['RK', 'BK']:
                    pygame.draw.circle(self.window, KING_RED if piece == 'RK' else KING_BLUE,
                                       (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2 - 5)

        print(f"Current player: {'Red (You)' if self.current_player == 'R' else 'Blue (AI)'}")
        pygame.display.update()
    
    def valid_moves(self, row, col):
        moves = []
        piece = self.board[row][col]
        
        # Determine if the piece is a king
        is_king = piece in ['RK', 'BK']
        
        # Determine valid directions based on piece type
        if piece in ['R', 'RK']:
            directions = [(-1, -1), (-1, 1)]  # Red moves up
            if is_king:
                directions.extend([(1, -1), (1, 1)])  # Kings can move down too
        else:  # 'B' or 'BK'
            directions = [(1, -1), (1, 1)]  # Blue moves down
            if is_king:
                directions.extend([(-1, -1), (-1, 1)])  # Kings can move up too
        
        # Check for captures first (mandatory in checkers)
        capture_moves = []
        for drow, dcol in directions:
            new_row, new_col = row + drow, col + dcol
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                # Check if there's an opponent's piece
                if (piece in ['R', 'RK'] and self.board[new_row][new_col] in ['B', 'BK']) or \
                   (piece in ['B', 'BK'] and self.board[new_row][new_col] in ['R', 'RK']):
                    # Check if we can jump over it
                    jump_row, jump_col = new_row + drow, new_col + dcol
                    if 0 <= jump_row < 8 and 0 <= jump_col < 8 and self.board[jump_row][jump_col] == '-':
                        capture_moves.append((jump_row, jump_col))
        
        # If captures are available, they are mandatory
        if capture_moves:
            return capture_moves
        
        # If no captures, regular moves
        for drow, dcol in directions:
            new_row, new_col = row + drow, col + dcol
            if 0 <= new_row < 8 and 0 <= new_col < 8 and self.board[new_row][new_col] == '-':
                moves.append((new_row, new_col))
        
        return moves
    
    def get_all_moves(self, board, player):
        all_moves = []
        # Check for any capture moves first 
        has_captures = False
        
        for row in range(8):
            for col in range(8):
                if board[row][col] in [player, player + 'K']:
                    moves = self.get_piece_moves(board, row, col)
                    if moves:
                        # Check if any of these moves are captures
                        for move_row, move_col in moves:
                            if abs(move_row - row) == 2: 
                                if not has_captures:
                                    all_moves = []  # Clear non-capture moves
                                    has_captures = True
                                all_moves.append(((row, col), (move_row, move_col)))
                                break 
                        
                       
                        if not has_captures:
                            for move_row, move_col in moves:
                                all_moves.append(((row, col), (move_row, move_col)))
        
        return all_moves
    
    def get_piece_moves(self, board, row, col):
        moves = []
        piece = board[row][col]
        
        # Similar to valid_moves but works on any board state
        is_king = piece in ['RK', 'BK']
        
        if piece in ['R', 'RK']:
            directions = [(-1, -1), (-1, 1)]
            if is_king:
                directions.extend([(1, -1), (1, 1)])
        else:  
            directions = [(1, -1), (1, 1)]
            if is_king:
                directions.extend([(-1, -1), (-1, 1)])
        
        # Check for captures first
        capture_moves = []
        for drow, dcol in directions:
            new_row, new_col = row + drow, col + dcol
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                # Check if there's an opponents piece
                if (piece in ['R', 'RK'] and board[new_row][new_col] in ['B', 'BK']) or \
                   (piece in ['B', 'BK'] and board[new_row][new_col] in ['R', 'RK']):
                    # Check if we can jump over it
                    jump_row, jump_col = new_row + drow, new_col + dcol
                    if 0 <= jump_row < 8 and 0 <= jump_col < 8 and board[jump_row][jump_col] == '-':
                        capture_moves.append((jump_row, jump_col))
        
        if capture_moves:
            return capture_moves
        
       #regular moves
        for drow, dcol in directions:
            new_row, new_col = row + drow, col + dcol
            if 0 <= new_row < 8 and 0 <= new_col < 8 and board[new_row][new_col] == '-':
                moves.append((new_row, new_col))
        
        return moves
    
    def simulate_move(self, board, move):
        """Apply a move to a board copy without changing the actual game state"""
        start, end = move
        start_row, start_col = start
        end_row, end_col = end
        
        # Create a copy of the board
        new_board = board.copy()
        
        # Move the piece
        new_board[end_row][end_col] = new_board[start_row][start_col]
        new_board[start_row][start_col] = '-'
        
        # Handle capture
        if abs(end_row - start_row) == 2:
            mid_row, mid_col = (start_row + end_row) // 2, (start_col + end_col) // 2
            new_board[mid_row][mid_col] = '-'
        
        # Handle promotion
        if end_row == 0 and new_board[end_row][end_col] == 'R':
            new_board[end_row][end_col] = 'RK'
        elif end_row == 7 and new_board[end_row][end_col] == 'B':
            new_board[end_row][end_col] = 'BK'
        
        return new_board
    
    def evaluate_board(self, board):
        #Evaluate the board position for the AI player (Blue)
        red_count = 0
        blue_count = 0
        red_kings = 0
        blue_kings = 0
        
        # Count pieces and their positions
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece == 'R':
                    red_count += 1
                    
                    red_count += (7 - row) * 0.1
                elif piece == 'B':
                    blue_count += 1
                    
                    blue_count += row * 0.1
                elif piece == 'RK':
                    red_kings += 3 
                elif piece == 'BK':
                    blue_kings += 3
        
        # Return the score (higher is better for Blue)
        return (blue_count + blue_kings) - (red_count + red_kings)
    
    def is_terminal_state(self, board):
        """Check if the game is over (no more valid moves or no pieces left)"""
        red_exists = False
        blue_exists = False
        
        for row in range(8):
            for col in range(8):
                if board[row][col] in ['R', 'RK']:
                    red_exists = True
                elif board[row][col] in ['B', 'BK']:
                    blue_exists = True
        
        if not red_exists or not blue_exists:
            return True
        
        # Check if any player has valid moves
        red_has_moves = False
        blue_has_moves = False
        
        for row in range(8):
            for col in range(8):
                if board[row][col] in ['R', 'RK']:
                    if self.get_piece_moves(board, row, col):
                        red_has_moves = True
                elif board[row][col] in ['B', 'BK']:
                    if self.get_piece_moves(board, row, col):
                        blue_has_moves = True
        
        return not (red_has_moves and blue_has_moves)
    
    def minimax(self, board, depth, maximizing_player):
        """Minimax algorithm without alpha-beta pruning"""

        self.nodes_expanded += 1
        # Base case: return evaluation if terminal state or depth is 0
        if depth == 0 or self.is_terminal_state(board):
            return self.evaluate_board(board)
        
        # Get all possible moves for current player
        player = 'B' if maximizing_player else 'R'
        all_moves = self.get_all_moves(board, player)
        
        # No moves left means game is over
        if not all_moves:
            return float('-inf') if maximizing_player else float('inf')
        
        if maximizing_player:
            best_value = float('-inf')
            for move in all_moves:
                new_board = self.simulate_move(board.copy(), move)
                value = self.minimax(new_board, depth - 1, False)
                best_value = max(best_value, value)
            return best_value
        else:
            best_value = float('inf')
            for move in all_moves:
                new_board = self.simulate_move(board.copy(), move)
                value = self.minimax(new_board, depth - 1, True)
                best_value = min(best_value, value)
            return best_value
    
    def minimax_alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        #Minimax algorithm with alpha-beta pruning
        # Base case: return evaluation if terminal state or depth is 0
        self.nodes_expanded += 1

        if depth == 0 or self.is_terminal_state(board):
            return self.evaluate_board(board)
        
        # Get all possible moves for current player
        player = 'B' if maximizing_player else 'R'
        all_moves = self.get_all_moves(board, player)
        
        # No moves left means game is over
        if not all_moves:
            return float('-inf') if maximizing_player else float('inf')
        
        if maximizing_player:
            best_value = float('-inf')
            for move in all_moves:
                new_board = self.simulate_move(board.copy(), move)
                value = self.minimax_alpha_beta(new_board, depth - 1, alpha, beta, False)
                best_value = max(best_value, value)
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    self.pruning_count += 1
                    break  # Beta cutoff
            return best_value
        else:
            best_value = float('inf')
            for move in all_moves:
                new_board = self.simulate_move(board.copy(), move)
                value = self.minimax_alpha_beta(new_board, depth - 1, alpha, beta, True)
                best_value = min(best_value, value)
                beta = min(beta, best_value)
                if beta <= alpha:
                    self.pruning_count += 1
                    break  # Alpha cutoff
            return best_value
    
    def ai_move(self):
        """Make the AI (Blue) player choose and perform a move"""
        if self.current_player != 'B':
            return
        
        # Get all possible moves for AI
        all_moves = self.get_all_moves(self.board, 'B')
        
        if not all_moves:
            print("Blue has no valid moves. Red wins!")
            pygame.quit()
            sys.exit()
        
        # Use the selected algorithm to find the best move
        best_move = None
        best_value = float('-inf')

        start_time = time.time()
        #reset the parameters
        self.nodes_expanded = 0
        self.pruning_count = 0
        # Show thinking animation
        self.draw_thinking_animation()
        
        if self.use_alpha_beta:
            # Alpha-Beta pruning with deeper search
            for move in all_moves:
                new_board = self.simulate_move(self.board.copy(), move)
                value = self.minimax_alpha_beta(new_board, self.depth, float('-inf'), float('inf'), False)
                if value > best_value:
                    best_value = value
                    best_move = move
        else:
            # Regular minimax
            for move in all_moves:
                new_board = self.simulate_move(self.board.copy(), move)
                value = self.minimax(new_board, self.depth, False)
                if value > best_value:
                    best_value = value
                    best_move = move
        end_time = time.time()
        self.last_move_time = end_time - start_time
        self.total_ai_time += self.last_move_time
        self.move_count += 1

        # Print performance metrics to console
        self.print_performance_metrics()
        
        # Apply the best move
        if best_move:
            start_row, start_col = best_move[0]
            end_row, end_col = best_move[1]
            
            # Move the piece
            self.board[end_row][end_col] = self.board[start_row][start_col]
            self.board[start_row][start_col] = '-'
            
            # Handle capture
            if abs(end_row - start_row) == 2:
                mid_row, mid_col = (start_row + end_row) // 2, (start_col + end_col) // 2
                self.board[mid_row][mid_col] = '-'
            
            # Handle promotion
            if end_row == 7 and self.board[end_row][end_col] == 'B':
                self.board[end_row][end_col] = 'BK'
            
            # Check for win
            if not any('R' in str(row) for row in self.board):
                print("Blue (AI) wins!")
                pygame.quit()
                sys.exit()
            
            # Switch player
            self.current_player = 'R'
            self.draw_board()

    def print_performance_metrics(self):
        """Print performance metrics to the console"""
        algorithm = "Alpha-Beta Pruning" if self.use_alpha_beta else "Regular Minimax"
        print("\n----- AI Move Performance Metrics -----")
        print(f"Algorithm: {algorithm} (Depth: {self.depth})")
        print(f"Execution time: {self.last_move_time:.4f} seconds")
        print(f"Nodes expanded: {self.nodes_expanded}")
        
        if self.use_alpha_beta:
            print(f"Branches pruned: {self.pruning_count}")
            pruning_efficiency = (self.pruning_count / max(1, self.nodes_expanded)) * 100
            print(f"Pruning efficiency: {pruning_efficiency:.2f}%")
        
        if self.move_count > 1:
            print(f"Average move time: {self.total_ai_time / self.move_count:.4f} seconds")
        
        print("---------------------------------------")
    
    def draw_thinking_animation(self):
        for i in range(3):
            dots = "." * (i + 1)
            print(f"AI is thinking{dots}")
            pygame.time.delay(300)
    
    def move_piece(self, row, col):
        """Handle player's move attempts"""
        if self.current_player != 'R':  # Only allow player to move Red pieces
            return
            
        if self.selected_piece:
            old_row, old_col = self.selected_piece
            if (row, col) in self.valid_moves(old_row, old_col):
                self.board[row][col] = self.board[old_row][old_col]
                self.board[old_row][old_col] = '-'
                
                # Capture move
                if abs(row - old_row) == 2:
                    mid_row, mid_col = (row + old_row) // 2, (col + old_col) // 2
                    self.board[mid_row][mid_col] = '-'
                
                # King promotion
                if row == 0 and self.board[row][col] == 'R':
                    self.board[row][col] = 'RK'
                
                # Check for win condition
                if not any('B' in str(row) for row in self.board):
                    print("Red (Player) wins!")
                    pygame.quit()
                    sys.exit()
                
                # Switch player and let AI make a move
                self.current_player = 'B'
                self.draw_board()
                pygame.time.delay(500)  
                self.ai_move()
            self.selected_piece = None
        else:
            if self.board[row][col] in ['R', 'RK']:
                # Check if the piece has valid moves before selecting it
                if self.valid_moves(row, col):
                    self.selected_piece = (row, col)


if __name__ == "__main__":
    # Ask user for AI settings
    print("Welcome to Checkers with AI!")
    print("1. Basic AI (Minimax)")
    print("2. Advanced AI (Minimax with Alpha-Beta Pruning)")
    choice = input("Choose AI type (1-2): ")
    use_alpha_beta = choice == '2'
    
    if use_alpha_beta:
        depth = int(input("Enter AI difficulty level (1-5, higher is harder): "))
    else:
        depth = int(input("Enter AI difficulty level (1-3, higher is harder): "))
    
    # Initialize game
    game = Checkers(use_alpha_beta=use_alpha_beta, depth=depth)
    
    # Main game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                row, col = y // CELL_SIZE, x // CELL_SIZE
                game.move_piece(row, col)
                game.draw_board()
        
        
        pygame.time.delay(50)
    
    pygame.quit()