import random
from typing import Tuple

import numpy as np

from drl_sample_project_python.drl_lib.do_not_touch.contracts import SingleAgentEnv


class GridWorld(SingleAgentEnv):
    def __init__(self, board: Tuple = (5, 5)):
        self.board = board
        self.nb_cells = board[0] * board[1]
        self.current_cell = 0
        self.step_count = 0
        self.rewards = [-1.0, 0.0, 1.0]
        self.global_actions = np.array([0, 1, 2, 3])

    def state_id(self) -> int:
        return self.current_cell

    def is_game_over(self) -> bool:
        if (self.current_cell == self.board[0] - 1 or
                self.current_cell == self.nb_cells - 1):
            return True
        else:
            return False

    def act_with_action_id(self, action_id: int):
        # TODO think about using this function and don't move current_cell if agent can't move in this direction
        self.step_count += 1
        # actions_reaction = [1, -1, self.board[0], - self.board[0]]
        if action_id == 0:  # left
            self.current_cell -= 1
        elif action_id == 1:  # right
            self.current_cell += 1
        elif action_id == 2:  # up
            self.current_cell -= self.board[0]
        elif action_id == 3:  # down
            self.current_cell += self.board[0]

    def score(self) -> float:
        if self.current_cell == self.board[0] - 1:
            return -1.0
        elif self.current_cell == self.nb_cells - 1:
            return 1.0
        else:
            return 0.0

    def available_actions_ids(self) -> np.ndarray:
        actions = [0, 1, 2, 3]
        # mur a gauche :
        if self.current_cell % self.board[0] == 0:
            actions.pop(0)
        # mur a droite :
        if self.current_cell % self.board[0] == self.board[0] - 1:
            actions.pop(1)
        # mur en haut :
        if self.current_cell < self.board[0]:
            actions.pop(2)
        # mur a down :
        if self.current_cell >= self.nb_cells - self.board[0]:
            actions.pop(3)

        return np.array(actions)

    def reset(self):
        self.current_cell = 0
        self.step_count = 0

    def view(self):
        print(f'Game Over: {self.is_game_over()}')
        print(f'score : {self.score()}')
        for i in range(self.board[0]):
            for j in range(self.board[1]):
                if i * self.board[0] + j == self.current_cell:
                    print("X", end='')
                else:
                    print("_", end='')
                if j < self.board[0] - 1:
                    print("", end="|")
            print()
        print()

    def reset_random(self):
        self.current_cell = random.randint(0, self.nb_cells - 1)
        self.step_count = 0

    def initialize_p(self, S, A, R):
        p = np.zeros((len(S), len(A), len(S), len(R)))

        for s in S[0: -1]:
            if s == self.board[0] - 1:
                continue

            p[s, 0, s - 1, 1] = 1.0

            if s == self.board[0] - 2:
                p[s, 1, s + 1, 0] = 1.0
            elif s == self.nb_cells - 2:
                p[s, 1, s + 1, 2] = 1.0
            else:
                p[s, 1, s + 1, 1] = 1.0

            if s > self.board[0] - 1:
                if self.current_cell == self.board[0] * 2 - 1:
                    p[s, 2, s - self.board[0], 0] = 1.0
                else:
                    p[s, 2, s - self.board[0], 1] = 1.0

            if s < self.nb_cells - self.board[0] - 1:
                if s == self.nb_cells - self.board[0] - 1:
                    p[s, 3, s + self.board[0], 2] = 1.0
                else:
                    if s < self.nb_cells - self.board[0] - 1:
                        p[s, 3, s + self.board[0], 1] = 1.0

        return p

    def create_policy(self):
        # return np.array([
        #     [0.0, 0.5, 0.0, 0.5],
        #     [0.33, 0.33, 0.0, 0.33],
        #     [0.5, 0.0, 0.0, 0.5],
        #
        #     [0.0, 0.33, 0.33, 0.33],
        #     [0.25, 0.25, 0.25, 0.25],
        #     [0.33, 0.0, 0.33, 0.33],
        #
        #     [0.0, 0.5, 0.5, 0.0],
        #     [0.33, 0.33, 0.33, 0.0],
        #     [0.5, 0.0, 0.5, 0.0]
        # ])
        # return np.array([
        #     [0, 0.0, 0, 1.0],
        #     [0.0, 0.0, 0, 1.0],
        #     [1.0, 0, 0, 1.0],
        #
        #     [0, 0.0, 0, 1.0],
        #     [0.0, 0.0, 0, 1.0],
        #     [1.0, 0, 0, 1.0],
        #
        #     [0, 1.0, 0.0, 0],
        #     [0.0, 1.0, 0.0, 0],
        #     [0.5, 0, 0.5, 0]
        # ])

        states_actions = (self.nb_cells, len(self.global_actions))
        init = np.ones(states_actions) * 0.25
        return init
