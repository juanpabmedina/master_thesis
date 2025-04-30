import random
import time
from typing import Tuple, Dict, List
from IPython.display import clear_output

# Constants
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
ACTION_TO_DELTA = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1),
    'STAY': (0, 0)
}

class SoccerGame:
    def __init__(self):
        self.rows = 4
        self.cols = 7  # 5 playable columns + 2 goal columns
        self.reset()

    def reset(self):
        self.player_positions = {'A': (1, 4), 'B': (2, 2)}
        self.ball_possession = random.choice(['A', 'B'])
        self.done = False
        self.scores = {'A': 0, 'B': 0}
        return self.get_state()

    def get_state(self):
        return (self.player_positions['A'], self.player_positions['B'], self.ball_possession)

    def _move(self, position, action):
        delta = ACTION_TO_DELTA[action]
        new_row = min(max(position[0] + delta[0], 0), self.rows - 1)
        new_col = position[1] + delta[1]

        # Restrict movement in the x corners (positions [0][0] and [3][0], [0][6] and [3][6])
        if new_col < 0 or new_col > 6:
            new_col = position[1]  # Prevent moving outside grid

        if (new_row == 0 or new_row == 3) and (new_col == 0 or new_col == 6):
            return position  # Prevent entering the x corners

        return (new_row, new_col)

    def step(self, actions: Dict[str, str]) -> Tuple[Tuple, Dict[str, int], bool]:
        order = ['A', 'B']
        random.shuffle(order)

        new_positions = self.player_positions.copy()
        for agent in order:
            other_agent = 'B' if agent == 'A' else 'A'
            new_pos = self._move(new_positions[agent], actions[agent])
            if new_pos == new_positions[other_agent]:
                self.ball_possession = other_agent
                continue
            new_positions[agent] = new_pos

        self.player_positions = new_positions

        reward = {'A': 0, 'B': 0}
        ball_holder = self.ball_possession
        ball_pos = self.player_positions[ball_holder]

        # Goal condition: ball enters the G positions (column 0 or 6) in rows 1 or 2
        if ball_holder == 'A' and ball_pos[1] == 6 and ball_pos[0] in [1, 2]:
            reward['A'] = 1
            reward['B'] = -1
            self.done = True
        elif ball_holder == 'B' and ball_pos[1] == 0 and ball_pos[0] in [1, 2]:
            reward['A'] = -1
            reward['B'] = 1
            self.done = True

        self.scores['A'] += reward['A']
        self.scores['B'] += reward['B']

        return self.get_state(), reward, self.done

    def print_state(self):
        grid = [[' . ' for _ in range(self.cols)] for _ in range(self.rows)]

        for r in [1, 2]:
            grid[r][0] = ' G '
            grid[r][6] = ' G '
        for r in [0, 3]:
            grid[r][0] = ' x '
            grid[r][6] = ' x '

        for player, pos in self.player_positions.items():
            grid[pos[0]][pos[1]] = ' ' + player + ('o' if self.ball_possession == player else ' ')

        clear_output(wait=True)
        for row in grid:
            print(''.join(row))
        print(f"Scores: {self.scores}\n")
        time.sleep(0.5)
