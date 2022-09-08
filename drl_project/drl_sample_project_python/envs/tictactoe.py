from drl_sample_project_python.drl_lib.do_not_touch.result_structures import PolicyAndActionValueFunction
from drl_sample_project_python.drl_lib.do_not_touch.contracts import SingleAgentEnv
# from typing import List

from tqdm import tqdm
import numpy as np
import random

P1 = 1
P2 = 2

class TicTacToe(SingleAgentEnv):
    def __init__(self, policy=None):

        self.board = np.zeros(9, dtype=int)
        self.icons = ["_", "X", "O"]
        self.state_id = 0
        self.rewards = [-1.0, 0.0, 1.0]  # lose egality win
        self.global_actions = np.array([x for x in range(9)])
        self.first_to_play = 1

        self.policy = policy if policy is not None else []
        self.S = all_states_ttt()

        self.current_score = 0

    def state_id(self) -> int:
        return self.state_id

    def new_state_id(self):
        flat_grid = [str(cell) for cell in self.board.tolist()]
        return ''.join(flat_grid)
        #return self.board.tolist()

    def is_game_over(self) -> bool:
        s_b = self.board
        for x in range(9):
            if x < 7 and s_b[x] != 0 and s_b[x] == s_b[x + 1] and s_b[x + 1] == s_b[x + 2] and x % 3 == 0:  # line
                return True
            if x < 3 and s_b[x] != 0 and s_b[x] == s_b[x + 3] and s_b[x + 3] == s_b[x + 6]:  # column
                return True
            if x == 0 and s_b[x] != 0 and s_b[x] == s_b[x + 4] and s_b[x + 4] == s_b[x + 8]:  # diag right
                return True
            if x == 2 and s_b[x] != 0 and s_b[x] == s_b[x + 2] and s_b[x + 2] == s_b[x + 4]:  # diag left
                return True
            if len(self.available_actions_ids()) == 0:
                return True # draw
        return False

    def act_with_action_id(self, action_id: int):
        a_as = self.available_actions_ids()
        if action_id < 9 and len(a_as) > 0 and action_id in a_as:
            self.board[action_id] = 1
            self.score()

        if len(self.available_actions_ids()) != 0:  # instant reply from opponent
            self.board[np.random.choice(self.available_actions_ids())] = 2
            self.score()

    def score(self) -> float:
        if self.win() == 1:
            self.current_score = 10
            return self.rewards[0]
        elif self.win() == 2:
            self.current_score = -10
            return self.rewards[2]
        else:
            return self.rewards[1]

    def win(self) -> int:
        s_b = self.board
        if self.is_game_over():
            for x in range(9):
                if x < 7 and s_b[x] != 2 and s_b[x] != 0 and s_b[x] == s_b[x + 1] and s_b[x + 1] == s_b[
                    x + 2] and x % 3 == 0:
                    return 1
                if x < 3 and s_b[x] != 2 and s_b[x] != 0 and s_b[x] == s_b[x + 3] and s_b[x + 3] == s_b[
                    x + 6]:
                    return 1
                if x == 0 and s_b[x] != 2 and s_b[x] != 0 and s_b[x] == s_b[x + 4] and s_b[x + 4] == s_b[
                    x + 8]:
                    return 1
                if x == 2 and s_b[x] != 2 and s_b[x] != 0 and s_b[x] == s_b[x + 2] and s_b[x + 2] == s_b[
                    x + 4]:
                    return 1
                if x < 7 and s_b[x] != 1 and s_b[x] != 0 and s_b[x] == s_b[x + 1] and s_b[x + 1] == s_b[
                    x + 2] and x % 3 == 0:
                    return 2
                if x < 3 and s_b[x] != 1 and s_b[x] != 0 and s_b[x] == s_b[x + 3] and s_b[x + 3] == s_b[
                    x + 6]:
                    return 2
                if x == 0 and s_b[x] != 1 and s_b[x] != 0 and s_b[x] == s_b[x + 4] and s_b[x + 4] == s_b[
                    x + 8]:
                    return 2
                if x == 2 and s_b[x] != 1 and s_b[x] != 0 and s_b[x] == s_b[x + 2] and s_b[x + 2] == s_b[
                    x + 4]:
                    return 2
        else:
            return 3

    def available_actions_ids(self):
        s_b = self.board
        available_action = []
        for index, x in enumerate(s_b):
            if x == 0:
                available_action.append(index)
        return available_action

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        self.icons = ["_", "X", "O"]
        self.current_score = 0

    def reset_random(self):
        # TODO
        self.reset()

    def view(self):
        for y in range(9):
            print("|", end="")
            print(f"{self.icons[self.board[y]]}|", sep="", end="")
            if y == 2 or y == 5:
                print()
        print()

    def create_policy(self):
        states_actions = (9, 9)
        init = np.ones(states_actions) * 0.11
        return init

    def agent_vs_human(self, ai_playing=False):
        self.view()
        if not ai_playing:
            while True:
                play = int(input("case à jouer (de 0 à 8): "))
                if play in self.available_actions_ids():
                    self.board[play] = 2
                    break
                else:
                    print("La case numéro ", play, " est déjà jouée. Veuillez en choisir une autre")
                    self.view()
        else:
            self.play_from_policy()

    def agent_vs_random(self, ai_playing=False):
        """
        Simule une partie entre un agent suivant une policy pi contre un adversaire jouant de manière uniformément aléatoire
        Lorsque ai_playing vaut False, le joueur random joue en premier
        Lorsque ai_playing faut True, l'agent joue en premier

        :param ai_playing:
        :return:
        """
        self.view()
        if not ai_playing:
            while True:
                play = np.random.choice(self.available_actions_ids())
                self.board[play] = 2
                print("Le joueur aléatoire joue case", play)
                break
        else:
            self.play_from_policy()

    def human_vs_random(self, ai_playing=False):
        self.view()
        if not ai_playing:
            while True:
                play = int(input("case à jouer (de 0 à 8): "))
                if play in self.available_actions_ids():
                    self.board[play] = 1
                    break
                else:
                    print("La case numéro ", play, " est déjà jouée. Veuillez en choisir une autre")
                    self.view()
        else:
            play = np.random.choice(self.available_actions_ids())
            self.board[play] = 2
            print("Le joueur aléatoire joue case", play)

    def game_human(self):
        ai_playing = self.first_to_play == 1
        while not self.is_game_over() and self.available_actions_ids():
            self.agent_vs_human(ai_playing=ai_playing)
            ai_playing = not ai_playing

        self.view()
        if self.win() == 2:
            print("Vous avez gagné")
        elif self.win() == 1:
            print("L'IA a gagné")
        else:
            print("Match nul")

    def game_random(self):
        ai_playing = self.first_to_play == 1 # TRUE 
        while not self.is_game_over() and self.available_actions_ids():
            self.agent_vs_random(ai_playing=ai_playing) # TRUE
            ai_playing = not ai_playing

        if self.win() == 2:
            print("Le joueur aléatoire a gagné")
        elif self.win() == 1:
            print("L'IA a gagné")
        else:
            print("Match nul")
        #self.view()

    def game_human_random(self):
        ai_playing = self.first_to_play == 1 # TRUE
        while not self.is_game_over() and self.available_actions_ids():
            self.human_vs_random(ai_playing=ai_playing) # TRUE
            ai_playing = not ai_playing

        if self.win() == 2:
            print("Le joueur aléatoire a gagné")
        elif self.win() == 1:
            print("Vous avez gagné")
        else:
            print("Match nul")
        #self.view()

    def play_from_policy_bis(self):
        # Problème : L'IA détermine la meilleure case sur laquelle jouer alors que la case est déjà remplie
        pi = self.policy
        s = self.new_state_id()
        pi_s = list(pi.item().values())[int(s)]
        choice = max(pi_s, key=pi_s.get)
        self.board[choice] = 1
        print("L'IA joue case ", choice)

    def play_from_policy(self):
        # Problème : L'IA détermine la meilleure case sur laquelle jouer alors que la case est déjà remplie
        index = self.S.index(self.board.tolist())
        choice = np.argmax(self.policy[index])
        self.board[choice] = 1
        print("L'IA joue case ", choice)


def playing():
    pi, q = load_policy_and_avf('../policies_tictactoe/policy_and_avf_offP_MC_control_26_06_23_05.npz')
    ttt = TicTacToe(pi)
    ttt.game_human()
    pass


def all_states_ttt():
    S = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    for e in range(3):
                        for f in range(3):
                            for g in range(3):
                                for h in range(3):
                                    for i in range(3):
                                        S.append([a, b, c, d, e, f, g, h, i])
    return S


def load_policy_and_avf(path):
    pi_avf = np.load(path)
    pi = pi_avf['pi']
    q = pi_avf['avf']

    return pi, q


def get_win_ratio_after_x_games(pi_avf: PolicyAndActionValueFunction, nb_games=100):
    # 0-D array with dtype object with a single element dict
    pi = pi_avf['pi']

    #print(list(pi.item().values())[000000000])
    #print(max(list(pi.item().values())[000000000], key=list(pi.item().values())[000000000].get))  # returns 'd'

    # 1: Humain-random // 2: IA
    players = np.array([1, 2])
    first_player = []
    wins = 0
    loses = 0
    draws = 0

    env = TicTacToe(pi)

    for _ in tqdm(range(nb_games)):
        # play a game following pi policy vs random player
        # self.first_to_play == 1 -> true // self.first_to_play == 2 -> false
        # false -> random play first // true -> ai agent play first
        # 1 ai agent play first // 2 random play first

        fp = np.random.choice(players)
        first_player.append(fp)
        env.first_to_play = fp
        env.game_random()

        if env.win() == 2:
            wins += 1
        elif env.win() == 1:
            loses += 1
        else:
            draws += 1
        env.reset()

    first_player = np.array(first_player)
    print("L'agent IA a commencé ", np.count_nonzero(first_player == 1), " parties")
    print("Le joueur random a commencé ", np.count_nonzero(first_player == 2), " parties")

    print("Le pourcentage de victoire de l'agent IA est de : ", (wins/(wins+loses+draws))*100, "%")
    print("Victoires : ", wins, " Défaites : ", loses, " Egalités : ", draws)
    return


if __name__ == '__main__':
    # Jouer avec une policy préalablement enregistrée
    #path = '../policies_tictactoe/policy_and_avf_offP_MC_control_28_06_00_50.npz'
    path = '../policies_tictactoe/policy_and_avf_offP_MC_control_28_06_02_51.npz'
    pi_avf = np.load(path, allow_pickle=True)
    print(get_win_ratio_after_x_games(pi_avf, 150))

    # Jouer (interaction humaine) avec random
    env = TicTacToe()
    env.game_human_random()





