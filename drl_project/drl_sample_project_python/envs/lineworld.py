import math
import numpy as np
from drl_sample_project_python.drl_lib.do_not_touch.contracts import MDPEnv


class LineWorld(MDPEnv):
    def __init__(self, nb_cells: int = 7):
        self.nb_cells = nb_cells
        self.current_cell = math.floor(nb_cells / 2)
        self.step_count = 0
        self.p = self.initialize_p(self.states(), self.actions(), self.rewards())
        self.v = self.initialize_V()

    def states(self) -> np.ndarray:
        return np.array([i for i in range(self.nb_cells)])

    def actions(self) -> np.ndarray:
        return np.array([0, 1])

    def rewards(self) -> np.ndarray:
        return np.array([-1.0, 0.0, 1.0])

    def is_state_terminal(self, s: int) -> bool:
        return self.current_cell in [0, self.nb_cells - 1]

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        return self.p[s, a, s_p, r] * (self.rewards()[r] + 0.999 * self.v[s_p])

    def view_state(self, s: int):
        for i in range(self.nb_cells):
            if i == self.current_cell:
                print("X", end='')
            else:
                print("_", end='')
        print()

    def initialize_V(self):
        V = np.random.random((self.nb_cells,))
        V[self.nb_cells - 1] = 0.0  # états finaux
        V[0] = 0.0
        return V

    def initialize_p(self, S, A, R):
        p = np.zeros((len(S), len(A), len(S), len(R)))

        for s in S[1:-1]:
            if s == 1:
                p[s, 0, s - 1, 0] = 1.0  # sur la case 1,  avec l'action '0' ( = aller à gauche) , le reward '0' ( de
                # valeur -1.0) vaut 1 (c'est le seul activé)
            else:
                p[s, 0, s - 1, 1] = 1.0

            if s == self.nb_cells - 2:
                p[s, 1, s + 1, 2] = 1.0  # same as l.22 , mais avec le reward de gain (valeur = 1.0)
            else:
                p[s, 1, s + 1, 1] = 1.0

        return p

    def create_policy(self, mode="random"):
        states_actions = (self.nb_cells, len(self.actions()))
        if mode == "left":
            pi = np.zeros(states_actions)
            pi[:, 0] = 1.0  # Pour toutes les cases , l'action '0' (= aller à gauche) vaut 1. (on la choisi)
        elif mode == "right":
            pi = np.zeros(states_actions)
            pi[:, 1] = 1.0  # pour toutes les cases, l'action '1' (= aller à droite) est activée
        else:
            pi = np.ones(states_actions) * 0.5

        return pi
