import numpy as np
import pygame
import sys
import time

pygame.init()

WIDTH, HEIGHT = 600, 600
LINE_WIDTH = 10
BOARD_ROWS, BOARD_COLS = 4, 4
CELL_SIZE = WIDTH // BOARD_COLS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

class TicTacToe:
    def __init__(self, game_mode="human_vs_human", ai_first=False, use_alpha_beta=False, max_depth=3):
        self.board = np.full((4, 4), '-')
        self.game_mode = game_mode
        
        if self.game_mode == "human_vs_ai":
            self.current_player = 'O' if ai_first else 'X'
            self.ai_first = ai_first
            self.use_alpha_beta = use_alpha_beta
            self.max_depth = max_depth
            self.nodes_expanded = 0
        else:
            # Human vs Human mode
            self.current_player = 'X'
            
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Tic-Tac-Toe 4x4")
        self.window.fill(WHITE)
        self.draw_grid()
        
        if self.game_mode == "human_vs_ai" and ai_first:
            self.ai_move()
    
    def draw_grid(self):
        for row in range(1, BOARD_ROWS):
            pygame.draw.line(self.window, BLACK, (0, row * CELL_SIZE), (WIDTH, row * CELL_SIZE), LINE_WIDTH)
        for col in range(1, BOARD_COLS):
            pygame.draw.line(self.window, BLACK, (col * CELL_SIZE, 0), (col * CELL_SIZE, HEIGHT), LINE_WIDTH)
        pygame.display.update()
    
    def draw_move(self, row, col):
        center_x = col * CELL_SIZE + CELL_SIZE // 2
        center_y = row * CELL_SIZE + CELL_SIZE // 2
        if self.current_player == 'X':
            pygame.draw.line(self.window, RED, (center_x - 50, center_y - 50), (center_x + 50, center_y + 50), LINE_WIDTH)
            pygame.draw.line(self.window, RED, (center_x + 50, center_y - 50), (center_x - 50, center_y + 50), LINE_WIDTH)
        else:
            pygame.draw.circle(self.window, BLACK, (center_x, center_y), 50, LINE_WIDTH)
        pygame.display.update()
    
    def make_move(self, row, col):
        if self.board[row, col] == '-':
            self.board[row, col] = self.current_player
            self.draw_move(row, col)
            
            if self.is_winner(self.current_player):
                if self.game_mode == "human_vs_ai":
                    winner = "AI" if self.current_player == 'O' else "Human"
                else:
                    winner = f"Player {self.current_player}"
                print(f"{winner} wins!")
                pygame.quit()
                sys.exit()
            elif self.is_full():
                print("It's a draw!")
                pygame.quit()
                sys.exit()
            
            # Switch players
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            
            # If in AI mode and it's AI's turn, make AI move
            if self.game_mode == "human_vs_ai" and self.current_player == 'O' and row is not None:
                self.ai_move()
    
    def is_winner(self, player):
        for row in self.board:
            if all(cell == player for cell in row):
                return True
        for col in range(4):
            if all(self.board[row][col] == player for row in range(4)):
                return True
        if all(self.board[i][i] == player for i in range(4)) or all(self.board[i][3 - i] == player for i in range(4)):
            return True
        return False
    
    def is_full(self):
        return '-' not in self.board
    
    def evaluate(self):
        if self.is_winner('X'):
            return 10
        if self.is_winner('O'):
            return -10
        return 0
    
    def minimax(self, depth, is_max):
        score = self.evaluate()
        
        if abs(score) == 10 or self.is_full() or depth >= self.max_depth:
            return score
        
        self.nodes_expanded += 1
        
        if is_max:
            best = float('-inf')
            for i in range(4):
                for j in range(4):
                    if self.board[i, j] == '-':
                        self.board[i, j] = 'X'
                        best = max(best, self.minimax(depth + 1, False))
                        self.board[i, j] = '-'
            return best
        else:
            best = float('inf')
            for i in range(4):
                for j in range(4):
                    if self.board[i, j] == '-':
                        self.board[i, j] = 'O'
                        best = min(best, self.minimax(depth + 1, True))
                        self.board[i, j] = '-'
            return best
    
    def minimax_alpha_beta(self, depth, is_max, alpha, beta):
        score = self.evaluate()
        
        if abs(score) == 10 or self.is_full() or depth >= self.max_depth:
            return score
        
        self.nodes_expanded += 1
        
        if is_max:
            best = float('-inf')
            for i in range(4):
                for j in range(4):
                    if self.board[i, j] == '-':
                        self.board[i, j] = 'X'
                        best = max(best, self.minimax_alpha_beta(depth + 1, False, alpha, beta))
                        self.board[i, j] = '-'
                        alpha = max(alpha, best)
                        if beta <= alpha:
                            break
            return best
        else:
            best = float('inf')
            for i in range(4):
                for j in range(4):
                    if self.board[i, j] == '-':
                        self.board[i, j] = 'O'
                        best = min(best, self.minimax_alpha_beta(depth + 1, True, alpha, beta))
                        self.board[i, j] = '-'
                        beta = min(beta, best)
                        if beta <= alpha:
                            break
            return best
    
    def ai_move(self):
        self.nodes_expanded = 0
        start_time = time.time()
        best_val = float('inf')
        best_move = (-1, -1)
        
        for i in range(4):
            for j in range(4):
                if self.board[i, j] == '-':
                    self.board[i, j] = 'O'
                    self.nodes_expanded += 1
                    move_val = self.minimax_alpha_beta(0, True, float('-inf'), float('inf')) if self.use_alpha_beta else self.minimax(0, True)
                    self.board[i, j] = '-'
                    if move_val < best_val:
                        best_val = move_val
                        best_move = (i, j)
        
        if best_move != (-1, -1):
            row, col = best_move
            self.board[row, col] = self.current_player
            self.draw_move(row, col)
            
            if self.is_winner(self.current_player):
                winner = "AI" if self.current_player == 'O' else "Human"
                print(f"{winner} wins!")
                pygame.quit()
                sys.exit()
            elif self.is_full():
                print("It's a draw!")
                pygame.quit()
                sys.exit()
                
            self.current_player = 'X'
        
        end_time = time.time()
        print(f"AI Move Time: {end_time - start_time:.4f} seconds, Nodes Expanded: {self.nodes_expanded}")

if __name__ == "__main__":
    print("Welcome to 4x4 Tic-Tac-Toe!")
    print("1. Play against AI")
    print("2. Play against another player")
    choice = input("Enter your choice (1/2): ")
    
    if choice == "1":
        # AI mode
        ai_first = input("Should AI play first? (yes/no): ").lower() == 'yes'
        use_alpha_beta = input("Use Alpha-Beta Pruning? (yes/no): ").lower() == 'yes'
        max_depth = 3
        game = TicTacToe(game_mode="human_vs_ai", ai_first=ai_first, use_alpha_beta=use_alpha_beta, max_depth=max_depth)
    else:
        # Human vs Human mode
        game = TicTacToe(game_mode="human_vs_human")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Only process mouse clicks for human players
                if game.game_mode == "human_vs_human" or game.current_player == 'X':
                    x, y = event.pos
                    row = y // CELL_SIZE
                    col = x // CELL_SIZE
                    game.make_move(row, col)
    
    pygame.quit()