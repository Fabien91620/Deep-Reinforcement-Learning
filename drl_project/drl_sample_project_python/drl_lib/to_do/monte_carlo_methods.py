import sys

import numpy

from drl_sample_project_python.drl_lib.do_not_touch.result_structures import PolicyAndActionValueFunction
from drl_sample_project_python.drl_lib.do_not_touch.single_agent_env_wrapper import Env2
from drl_sample_project_python.envs.tictactoe import TicTacToe, all_states_ttt

import numpy as np
from datetime import datetime
import random
from tqdm import tqdm
import json


def monte_carlo_es_on_tic_tac_toe_solo(iter_count):
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    TTT = TicTacToe()
    S = all_states_ttt()

    A = TTT.global_actions
    R = TTT.rewards

    pi = np.ones((len(S), len(A))) * (1 / 9)  # ppblty to play each action for each state
    # for s in S:
    #     pi[s] /= np.sum(pi[s])
    q = np.random.random((len(S), len(A)))

    Returns = [[[] for _ in A] for _ in S]

    for it in range(iter_count):
        s0 = random.randint(0, len(S) - 1)

        TTT.board = S[s0].copy()

        if TTT.is_game_over() or len(TTT.available_actions_ids()) == 0:
            continue
        a0 = np.random.choice(TTT.available_actions_ids())
        s = S[s0].copy()
        a = a0

        s_p, r, terminal = step(TTT, a0)
        s_history = [s]
        a_history = [a]
        s_p_history = [s_p]
        r_history = [r]

        step_count = 1
        while terminal is False and step_count < 100:
            s = s_p
            actions = TTT.available_actions_ids()
            if len(actions) == 0:
                break
            a = np.random.choice(actions)

            s_p, r, terminal = step(TTT, a)
            s_history.append(s)
            a_history.append(a)
            s_p_history.append(s_p)
            r_history.append(r)
            step_count += 1

        G = 0
        for t in reversed(range(len(s_history))):
            G = 0.999 * G + r_history[t]
            s_t = s_history[t]
            a_t = a_history[t]

            appear = False
            for t_p in range(t - 1):
                if s_history[t_p] == s_t and a_history[t_p] == a_t:
                    appear = True
                    break
            if appear:
                continue

            state = S.index(s_t)  # get id of situation at the end of the game

            Returns[state][a_t].append(G)
            q[state, a_t] = np.mean(Returns[state][a_t])
            pi[state, :] = 0.0
            argmax = np.argmax(q[state])
            pi[state, argmax] = 1.0

    TTT.view()
    return PolicyAndActionValueFunction(pi, q)


def monte_carlo_es_on_tic_tac_toe_solo_bis(iter_count):
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    TTT = TicTacToe()
    S = all_states_ttt()

    A = TTT.global_actions
    R = TTT.rewards

    pi = np.ones((len(S), len(A))) * (1 / 9)  # ppblty to play each action for each state
    # for s in S:
    #     pi[s] /= np.sum(pi[s])
    q = np.random.random((len(S), len(A)))

    Returns = [[[] for _ in A] for _ in S]

    for it in range(iter_count):
        s0 = random.randint(0, len(S) - 1)

        TTT.board = S[s0].copy()

        if TTT.is_game_over() or len(TTT.available_actions_ids()) == 0:
            continue
        a0 = np.random.choice(TTT.available_actions_ids())
        s = S[s0].copy()
        a = a0

        s_p, r, terminal = step(TTT, a0)
        s_history = [s]
        a_history = [a]
        s_p_history = [s_p]
        r_history = [r]

        step_count = 1
        while terminal is False and step_count < 100:
            s = s_p
            actions = TTT.available_actions_ids()
            if len(actions) == 0:
                break
            a = np.random.choice(actions)

            s_p, r, terminal = step(TTT, a)
            s_history.append(s)
            a_history.append(a)
            s_p_history.append(s_p)
            r_history.append(r)
            step_count += 1

        G = 0
        for t in reversed(range(len(s_history))):
            G = 0.999 * G + r_history[t]
            s_t = s_history[t]
            a_t = a_history[t]

            appear = False
            for t_p in range(t - 1):
                if s_history[t_p] == s_t and a_history[t_p] == a_t:
                    appear = True
                    break
            if appear:
                continue

            state = S.index(s_t)  # get id of situation at the end of the game

            Returns[state][a_t].append(G)
            q[state, a_t] = np.mean(Returns[state][a_t])
            pi[state, :] = 0.0
            argmax = np.argmax(q[state])
            pi[state, argmax] = 1.0

    TTT.view()
    return PolicyAndActionValueFunction(pi, q)


def step(tictactoe: TicTacToe, a: int) -> (list, float, bool):
    tictactoe.act_with_action_id(a)
    return tictactoe.board.copy(), tictactoe.score(), tictactoe.is_game_over()


def on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo(iter_count: int) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy
    and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    epsilon = 0.01
    TTT = TicTacToe()
    S = all_states_ttt()

    A = TTT.global_actions
    R = TTT.rewards

    pi = np.ones((len(S), len(A))) * (1 / 9)  # ppblty to play each action for each state
    # for s in S:
    #     pi[s] /= np.sum(pi[s])
    q = np.random.random((len(S), len(A)))

    Returns = [[[] for _ in A] for _ in S]

    for it in range(iter_count):
        s0 = random.randint(0, len(S) - 1)

        TTT.board = S[s0].copy()

        if TTT.is_game_over() or len(TTT.available_actions_ids()) == 0:
            continue
        a0 = np.random.choice(TTT.available_actions_ids())
        s = S[s0].copy()
        a = a0

        s_p, r, terminal = step(TTT, a0)
        s_history = [s]
        a_history = [a]
        s_p_history = [s_p]
        r_history = [r]

        step_count = 1
        while terminal is False and step_count < 100:
            s = s_p
            actions = TTT.available_actions_ids()
            if len(actions) == 0:
                break
            a = np.random.choice(actions)

            s_p, r, terminal = step(TTT, a)
            s_history.append(s)
            a_history.append(a)
            s_p_history.append(s_p)
            r_history.append(r)
            step_count += 1

        G = 0
        for t in reversed(range(len(s_history))):
            G = 0.999 * G + r_history[t]
            s_t = s_history[t]
            a_t = a_history[t]

            appear = False
            for t_p in range(t - 1):
                if s_history[t_p] == s_t and a_history[t_p] == a_t:
                    appear = True
                    break
            if appear:
                continue

            state = S.index(s_t)  # get id of situation at the end of the game

            Returns[state][a_t].append(G)
            q[state, a_t] = np.mean(Returns[state][a_t])

            TTT.board = S[state]
            argmax = np.argmax(q[state])
            pi[state, :] = epsilon / len(TTT.available_actions_ids())
            pi[state, argmax] = 1 - epsilon + epsilon / len(TTT.available_actions_ids())

    TTT.view()
    return PolicyAndActionValueFunction(pi, q)


def off_policy_monte_carlo_control_on_tic_tac_toe_solo(iter_count: int) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    TTT = TicTacToe()
    S = all_states_ttt()

    A = TTT.global_actions
    R = TTT.rewards

    # pi = np.ones((len(S), len(A))) * (1 / 9)  # ppblty to play each action for each state
    # for s in S:
    #     pi[s] /= np.sum(pi[s])
    q = np.random.random((len(S), len(A)))
    C = np.zeros((len(S), len(A)))
    pi = np.random.random((len(S), len(A)))
    for s in range(len(S)):
        pi[s, :] = 0.
        argmax = np.argmax(q[s])
        pi[s, argmax] = 1.0

    while iter_count > 0:
        b = np.ones((len(S), len(A))) * (1 / 9)

        s0 = random.randint(0, len(S) - 1)

        TTT.board = S[s0].copy()

        if TTT.is_game_over() or len(TTT.available_actions_ids()) == 0:
            continue
        a0 = np.random.choice(TTT.available_actions_ids())
        s = S[s0].copy()
        a = a0

        s_p, r, terminal = step(TTT, a0)
        s_history = [s]
        a_history = [a]
        s_p_history = [s_p]
        r_history = [r]

        step_count = 1
        while terminal is False and step_count < 100:
            s = s_p
            actions = TTT.available_actions_ids()
            if len(actions) == 0:
                break
            a = np.random.choice(actions)

            s_p, r, terminal = step(TTT, a)
            s_history.append(s)
            a_history.append(a)
            s_p_history.append(s_p)
            r_history.append(r)
            step_count += 1

        G = 0
        W = 1

        for step_id in reversed(range(1, len(s_history))):
            s_t = s_history[step_id]
            a_t = a_history[step_id]
            state = S.index(s_t)
            G = 0.999 * G + r_history[step_id]
            C[state, a_t] = C[state, a_t] + W

            q[state, a_t] = q[state, a_t] + (W / C[state, a_t]) * (G - q[state, a_t])

            pi[state, :] = 0.0
            argmax = np.argmax(q[state])
            pi[state, argmax] = 1.0

            if a_t != argmax:
                break

            W = W * (1 / b[state, a_t])
        iter_count -= 1

    return PolicyAndActionValueFunction(pi, q)


def off_policy_monte_carlo_control_on_tic_tac_toe_solo_bis(num_episodes: int, gamma=0.999) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """

    ttt = TicTacToe()
    Q = {}
    C = {}
    pi = {}
    b = {}

    for _ in tqdm(range(num_episodes)):
        ttt.reset()
        S = []
        A = []
        R = []

        episode = []
        while not ttt.is_game_over():
            s = str(ttt.new_state_id())
            S.append(s)
            available_actions = ttt.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                Q[s] = {}
                C[s] = {}
                b[s] = {}
                for a in available_actions:

                    pi[s][a]=1/len(available_actions)
                    Q[s][a] = 0.0
                    C[s][a] = 0.0
                    b[s][a] = 1/len(available_actions)
            #print(len(available_actions))
            # print(list(pi[s].values()))
            action = np.random.choice(a=available_actions, p=list(b[s].values()))
            #print(ttt.view())
            #print("game over avant action_id: ", ttt.is_game_over())
            A.append(action)

            old_r = ttt.current_score
            ttt.act_with_action_id(action)
            #print("action a jouer: ", action)
            #print("game over après action_id: ", ttt.is_game_over())
            r = ttt.current_score
            #print("old score: ", old_r, " new_score: ", r)
            R.append(r - old_r)

        G = 0.0
        W = 1.0
        for t in reversed(range(len(S))):
            G = gamma * G + R[t]
            C[S[t]][A[t]] = C[S[t]][A[t]] + W
            Q[S[t]][A[t]] += (W/C[S[t]][A[t]]) * (G - Q[S[t]][A[t]])

            max_a = None
            best_a = None
            for a in Q[S[t]]:
                if max_a is None or Q[S[t]][a] > max_a:
                    max_a = Q[S[t]][a]
                    best_a = a
            for a in pi[S[t]]:
                pi[S[t]][a] = 0
            pi[S[t]][best_a] = 1

            if A[t] != best_a:
                break
            W = W * 1/b[S[t]][A[t]]

    return PolicyAndActionValueFunction(pi, Q)


def monte_carlo_es_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = Env2()
    # TODO
    pass


def on_policy_first_visit_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    # TODO
    pass


def off_policy_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    # TODO
    pass


def save_policy_and_action_value_function_to_json(policy: PolicyAndActionValueFunction, algo_name: str):
    dt = datetime.now().strftime("%d_%m_%H_%M")
    path = '../../policies_tictactoe/policy_and_avf_' + algo_name + '_' + dt

    np.savez(path, pi=policy.pi, avf=policy.q, allow_pickle=True)


def save_policy_avf(policy: PolicyAndActionValueFunction, algo_name: str):
    dt = datetime.now().strftime("%d_%m_%H_%M")
    path = '../../policies_tictactoe/policy_and_avf_' + algo_name + '_' + dt

    with open(path, 'wb') as f:
        np.save(f, policy.pi)
        np.save(f, policy.q)

def demo():
    # print(monte_carlo_es_on_tic_tac_toe_solo(1000))
    # print(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo(1000))
    p_avf_offP_MC_control = off_policy_monte_carlo_control_on_tic_tac_toe_solo(20000)
    save_policy_and_action_value_function_to_json(p_avf_offP_MC_control, "offP_MC_control")
    #save_policy_avf(p_avf_offP_MC_control, "offP_MC_control")
    #p_avf_MC_ES = monte_carlo_es_on_tic_tac_toe_solo(10000)
    #save_policy_and_action_value_function_to_json(p_avf_MC_ES, "MC_ES")

    # print(monte_carlo_es_on_secret_env2())
    # print(on_policy_first_visit_monte_carlo_control_on_secret_env2())
    # print(off_policy_monte_carlo_control_on_secret_env2())


if __name__ == '__main__':
    #ttt = TicTacToe()
    #ttt.game()
    demo()
    # with open("drl_project/drl_sample_project_python\policies_tictactoe/policy_and_avf_offP_MC_control_26_06_22_52",'rb') as f:
    #     pi = np.load(f)
    #     q = np.load(f)
    # print(pi, q)
