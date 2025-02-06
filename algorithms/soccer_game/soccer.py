import random
from IPython.display import clear_output
import time


class SoccerGame:
    def __init__(self):
        self.grid_size = (4, 5)  # 4x5 grid
        self.reset_game()
        self.reward = 0

    def reset_pos(self):
        """Reset the game positions."""
        self.player_positions = {'A': (1, 3), 'B': (2, 1)}  # Initial positions
        self.ball_possession = random.choice(['A', 'B'])  # Random initial possession
        #self.scores = {'A': 0, 'B': 0}
        #print(f"Game reset. Ball possession: {self.ball_possession}")

    def reset_game(self):
        """Reset the game state."""
        self.player_positions = {'A': (1, 3), 'B': (2, 1)}  # Initial positions
        self.ball_possession = random.choice(['A', 'B'])  # Random initial possession
        self.scores = {'A': 0, 'B': 0}
        print(f"Game reset. Ball possession: {self.ball_possession}")

    def is_valid_position(self, pos):
        """Check if the position is within the grid."""
        return 0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]

    def move(self, player, action):
        """Calculate new position based on action."""
        x, y = self.player_positions[player]
        if action == 'N':  # Move North
            return (x - 1, y)
        elif action == 'S':  # Move South
            return (x + 1, y)
        elif action == 'E':  # Move East
            return (x, y + 1)
        elif action == 'W':  # Move West
            return (x, y - 1)
        elif action == 'stand':
            return (x, y)
        return (x, y)  # No move

    def play_turn(self, actions, invert_players = False):
        """Execute a turn with both players' actions."""
        moves = list(actions.items())
        random.shuffle(moves)  # Randomize turn order
        new_positions = self.player_positions.copy()

        for player, action in moves:
            new_pos = self.move(player, action)
            opponent = 'B' if player == 'A' else 'A'

            # Check for collisions or valid move
            if new_pos == self.player_positions[opponent]:  # Collision
                self.ball_possession = opponent
                new_positions[player] = self.player_positions[player]
                #print("Collision")
            elif self.is_valid_position(new_pos):
                new_positions[player] = new_pos
                #print("Valid move")
            else:  # Invalid move, stays in place
                new_positions[player] = self.player_positions[player]
                #print("Invaid move")

            self.player_positions = new_positions
        done = False
        # Handle scoring
        if self.ball_possession == 'A' and self.player_positions['A'] in [(2,0),(1,0)]:  # A scores
            self.scores['A'] += 1
            self.reset_pos()
            self.reward = {'A': 1, 'B': -1}
            done = True
        elif self.ball_possession == 'B' and self.player_positions['B'] in [(2,4),(1,4)]:  # B scores
            self.scores['B'] += 1
            self.reward = {'A': -1, 'B': 1}
            self.reset_pos()
            done = True
        else:
            self.reward = {'A': 0, 'B': 0}

        self.new_state = (self.player_positions['A'], self.player_positions['B'], self.ball_possession)

        return self.new_state, self.reward, done
    
    def print_state(self):
        """Display the grid and scores with movement effect in Jupyter Notebook."""
        # Crear la cuadrícula
        grid = [[' . ' for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        for player, pos in self.player_positions.items():
            grid[pos[0]][pos[1]] = ' ' + player + ('o' if self.ball_possession == player else ' ')
        
        # Limpiar la salida anterior
        clear_output(wait=True)  # Limpia la celda del notebook para simular movimiento
        
        # Mostrar el estado actual
        for row in grid:
            print(' '.join(row))
        print(f"Scores: {self.scores}")
        time.sleep(0.5)  # Agregar un pequeño retraso para simular movimiento
