import pygame
import numpy as np
import random
import time
import os

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600
BOARD_SIZE = 8
CELL_SIZE = WIDTH // BOARD_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
FONT_SIZE = 24

# Initialize font
pygame.font.init()
font = pygame.font.SysFont(None, FONT_SIZE)

# Initialize Pygame window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("8-Queens Problem")


try:
    QUEEN_IMAGE = pygame.image.load("queen.png")
    QUEEN_IMAGE = pygame.transform.scale(QUEEN_IMAGE, (CELL_SIZE, CELL_SIZE))
except pygame.error:
   
    QUEEN_IMAGE = None
    print("pic not found")

# Function to draw the board
def draw_board(board, algorithm_name="", conflicts=None):
    screen.fill(WHITE)
    
    # Draw the chessboard
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if (row + col) % 2 == 0:
                pygame.draw.rect(screen, BLACK, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            
            # Draw queens
            if board[col] == row:
                
                screen.blit(QUEEN_IMAGE, (col * CELL_SIZE, row * CELL_SIZE))

    
    pygame.display.update()

# Already given Board
def generate_initial_board():
    board = np.random.randint(0, BOARD_SIZE, BOARD_SIZE)
    print(f"Generated initial board: {board}")
    print(f"Initial conflicts: {get_conflicts(board)}")
    return board

# Conflict Calculation
def get_conflicts(board):
    conflicts = 0
    for col in range(BOARD_SIZE):
        for other_col in range(col + 1, BOARD_SIZE):
            if board[col] == board[other_col] or abs(board[col] - board[other_col]) == abs(col - other_col):
                conflicts += 1
    return conflicts

# Hill Climbing with Multiple Restarts
def hill_climbing_restarts(max_restarts=50, max_iterations=100):
    best_solution = None
    best_conflicts = float('inf')
    total_iterations = 0
    successful_restarts = 0
    start_time = time.time()
    
    print("\n===== HILL CLIMBING WITH RESTARTS =====")
    print(f"Parameters: Max Restarts={max_restarts}, Max Iterations per Restart={max_iterations}")
    
    for restart in range(max_restarts):
        board = generate_initial_board()
        print(f"\nRestart {restart+1}/{max_restarts}")
        print(f"Initial state: {board}")
        initial_conflicts = get_conflicts(board)
        print(f"Initial conflicts: {initial_conflicts}")
        
        draw_board(board, "Hill Climbing (Restart " + str(restart+1) + "/" + str(max_restarts) + ")", 
                   initial_conflicts)
        time.sleep(0.5)
        
        iterations = 0
        stuck = False
        
        while iterations < max_iterations and not stuck:
            iterations += 1
            total_iterations += 1
            current_conflicts = get_conflicts(board)
            best_move = None
            min_conflicts = current_conflicts
            
            for col in range(BOARD_SIZE):
                original_row = board[col]
                for row in range(BOARD_SIZE):
                    if row == original_row:
                        continue
                    board[col] = row
                    new_conflicts = get_conflicts(board)
                    if new_conflicts < min_conflicts:
                        min_conflicts = new_conflicts
                        best_move = (col, row)
                board[col] = original_row
            
            if best_move is None:
                print(f"Local minimum reached at iteration {iterations}")
                print(f"Conflicts at local minimum: {current_conflicts}")
                stuck = True
                break
                
            board[best_move[0]] = best_move[1]
            new_conflicts = get_conflicts(board)
            print(f"Iteration {iterations}: Move queen at column {best_move[0]} to row {best_move[1]} - Conflicts: {new_conflicts}")
            
            draw_board(board, "Hill Climbing (Restart " + str(restart+1) + "/" + str(max_restarts) + ")", 
                       new_conflicts)
            time.sleep(0.3)
            
            if new_conflicts == 0:
                print(f"Solution found at restart {restart+1}, iteration {iterations}!")
                print(f"Final state: {board}")
                successful_restarts += 1
                elapsed_time = time.time() - start_time
                print(f"Total time: {elapsed_time:.2f} seconds")
                print(f"Total iterations: {total_iterations}")
                return board
        
        final_conflicts = get_conflicts(board)
        print(f"End of restart {restart+1}: Conflicts = {final_conflicts}")
        if best_solution is None or final_conflicts < best_conflicts:
            best_solution = board.copy()
            best_conflicts = final_conflicts
            print(f"New best solution found: {best_solution} with {best_conflicts} conflicts")
    
    elapsed_time = time.time() - start_time
    print("\n===== HILL CLIMBING SUMMARY =====")
    print(f"Best solution found: {best_solution}")
    print(f"Conflicts in best solution: {best_conflicts}")
    print(f"Total iterations: {total_iterations}")
    print(f"Successful restarts: {successful_restarts}/{max_restarts}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    
    return best_solution

# Simulated Annealing Algorithm
def simulated_annealing(board, temperature=100, cooling_rate=0.98, max_iterations=5000):
    board = board.copy()
    iterations = 0
    moves_accepted = 0
    uphill_moves = 0
    start_time = time.time()
    initial_conflicts = get_conflicts(board)
    
    print("\n===== SIMULATED ANNEALING =====")
    print(f"Parameters: Initial Temperature={temperature}, Cooling Rate={cooling_rate}, Max Iterations={max_iterations}")
    print(f"Initial state: {board}")
    print(f"Initial conflicts: {initial_conflicts}")
    
    while temperature > 1 and iterations < max_iterations:
        iterations += 1
        col = random.randint(0, BOARD_SIZE - 1)
        row = random.randint(0, BOARD_SIZE - 1)
        original_row = board[col]
        
        # Skip if no change
        if original_row == row:
            continue
            
        old_conflicts = get_conflicts(board)
        board[col] = row
        new_conflicts = get_conflicts(board)
        
        delta = new_conflicts - old_conflicts
        
        if iterations % 100 == 0 or new_conflicts == 0:
            print(f"Iteration {iterations}: T={temperature:.2f}, Conflicts={new_conflicts}, Moves accepted={moves_accepted}, Uphill moves={uphill_moves}")
        
        # Correct acceptance criteria
        if delta < 0 or random.random() < np.exp(-delta / temperature):
            # Accept the new state
            moves_accepted += 1
            if delta > 0:
                uphill_moves += 1
                print(f"Iteration {iterations}: Accepted uphill move with delta={delta}, T={temperature:.2f}")
        else:
            # Revert to the original state
            board[col] = original_row
            continue
        
        draw_board(board, f"Simulated Annealing (T={temperature:.2f})", 
                  new_conflicts)
        time.sleep(0.3)
        
        temperature *= cooling_rate
        
        if new_conflicts == 0:
            elapsed_time = time.time() - start_time
            print(f"\nSolution found at iteration {iterations}!")
            print(f"Final state: {board}")
            print(f"Total moves accepted: {moves_accepted}")
            print(f"Uphill moves accepted: {uphill_moves}")
            print(f"Final temperature: {temperature:.4f}")
            print(f"Total time: {elapsed_time:.2f} seconds")
            return board
    
    elapsed_time = time.time() - start_time
    final_conflicts = get_conflicts(board)
    print("\n===== SIMULATED ANNEALING SUMMARY =====")
    print(f"Best solution found: {board}")
    print(f"Conflicts in best solution: {final_conflicts}")
    print(f"Total iterations: {iterations}")
    print(f"Moves accepted: {moves_accepted}")
    print(f"Uphill moves: {uphill_moves}")
    print(f"Final temperature: {temperature:.4f}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    
    return board

# Local Beam Search Algorithm
def local_beam_search(k=3, max_iterations=100):
    states = [generate_initial_board() for _ in range(k)]
    iterations = 0
    start_time = time.time()
    
    print("\n===== LOCAL BEAM SEARCH =====")
    print(f"Parameters: Beam Width (k)={k}, Max Iterations={max_iterations}")
    
    print("\nInitial states:")
    for i, state in enumerate(states):
        print(f"State {i+1}: {state} - Conflicts: {get_conflicts(state)}")
    
    while iterations < max_iterations:
        iterations += 1
        
        # Sort states by conflicts
        states.sort(key=get_conflicts)
        best_state = states[0]
        best_conflicts = get_conflicts(best_state)
        
        print(f"\nIteration {iterations}")
        print(f"Best state: {best_state}")
        print(f"Conflicts: {best_conflicts}")
        
        # Visualize the best state
        draw_board(best_state, f"Local Beam Search (Iteration {iterations})", 
                  best_conflicts)
        time.sleep(0.5)
        
        if best_conflicts == 0:
            elapsed_time = time.time() - start_time
            print(f"\nSolution found at iteration {iterations}!")
            print(f"Final state: {best_state}")
            print(f"Total time: {elapsed_time:.2f} seconds")
            return best_state
            
        # Generate successor states
        new_states = []
        successors_generated = 0
        
        for state in states[:k]:
            for col in range(BOARD_SIZE):
                original_row = state[col]
                for row in range(BOARD_SIZE):
                    if state[col] != row:
                        new_state = state.copy()
                        new_state[col] = row
                        new_states.append(new_state)
                        successors_generated += 1
        
        # Ensure we keep the best k states
        new_states.sort(key=get_conflicts)
        states = new_states[:k]
        
        print(f"Generated {successors_generated} successors")
        print(f"Selected top {k} states for next iteration:")
        for i, state in enumerate(states):
            print(f"  State {i+1}: Conflicts = {get_conflicts(state)}")
    
    # Return the best solution found
    states.sort(key=get_conflicts)
    best_solution = states[0]
    final_conflicts = get_conflicts(best_solution)
    
    elapsed_time = time.time() - start_time
    print("\n===== LOCAL BEAM SEARCH SUMMARY =====")
    print(f"Best solution found: {best_solution}")
    print(f"Conflicts in best solution: {final_conflicts}")
    print(f"Total iterations: {iterations}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    
    return best_solution

# Main function
def main():
    board = generate_initial_board()
    print(f"Starting with board: {board}")
    print(f"Initial conflicts: {get_conflicts(board)}")
    draw_board(board, "Initial Board", get_conflicts(board))
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    print("\nSolving using Hill Climbing with Restarts...")
                    board = hill_climbing_restarts()
                elif event.key == pygame.K_s:
                    print("\nSolving using Simulated Annealing...")
                    board = simulated_annealing(board)
                elif event.key == pygame.K_b:
                    print("\nSolving using Local Beam Search...")
                    board = local_beam_search()
                
                final_conflicts = get_conflicts(board)
                print(f"\nFinal board state: {board}")
                print(f"Final conflicts: {final_conflicts}")
                print("Success!" if final_conflicts == 0 else "Did not find perfect solution.")
                draw_board(board, "Final Solution", final_conflicts)
    
    pygame.quit()

if __name__ == "__main__":
    main()