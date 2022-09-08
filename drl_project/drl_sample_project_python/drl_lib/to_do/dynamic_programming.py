import numpy as np

from drl_sample_project_python.drl_lib.do_not_touch.mdp_env_wrapper import Env1
from drl_sample_project_python.drl_lib.do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction
from drl_sample_project_python.envs.gridworld import GridWorld
from drl_sample_project_python.envs.lineworld import LineWorld


def policy_evaluation_on_line_world(pi, lineWorld) -> ValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    theta = 0.0000001

    while True:
        delta = 0
        for state_id in lineWorld.states():
            v = lineWorld.v[state_id]
            lineWorld.v[state_id] = 0.0
            for a in lineWorld.actions():
                total = 0.0
                for s_p in lineWorld.states():
                    for r in range(len(lineWorld.rewards())):
                        # total += Bellman(p, state_id, a, s_p, r, R, V)
                        total += lineWorld.transition_probability(state_id, a, s_p, r)
                total *= pi[state_id, a]
                lineWorld.v[state_id] += total
            delta = max(delta, np.abs(v - lineWorld.v[state_id]))
        if delta < theta:
            break
    # lineWorld.view_state(0)
    return lineWorld.v


def Bellman(p, state_id, a, s_p, r, R, V):
    return p[state_id, a, s_p, r] * (R[r] + 0.999 * V[s_p])


def policy_iteration_on_line_world(pi, lineWorld: LineWorld) -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    theta = 0.0000001

    while True:
        lineWorld.v = policy_evaluation_on_line_world(pi, lineWorld)
        stable = True
        for s in lineWorld.states():
            old_pi_s = pi[s].copy()
            best_a = -1
            best_a_score = -99999999999
            for a in lineWorld.actions():
                total = 0
                for s_p in lineWorld.states():
                    for r in range(len(lineWorld.rewards())):
                        # total += Bellman(p, s, a, s_p, r, R, V)
                        total += lineWorld.transition_probability(s, a, s_p, r)

                if total > best_a_score:
                    best_a = a
                    best_a_score = total
            pi[s, :] = 0.0
            pi[s, best_a] = 1.0
            if np.any(pi[s] != old_pi_s):
                stable = False
            if stable:
                return PolicyAndValueFunction(pi, lineWorld.v)


def value_iteration_on_line_world(pi, lineWorld: LineWorld) -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    theta = 0.0000001

    while True:
        while True:
            delta = 0
            for state_id in lineWorld.states():
                v = lineWorld.v[state_id]
                lineWorld.v[state_id] = 0.0
                for a in lineWorld.actions():
                    total = 0.0
                    for s_p in lineWorld.states():
                        for r in range(len(lineWorld.rewards())):
                            # total += Bellman(p, state_id, a, s_p, r, R, V)
                            total += lineWorld.transition_probability(state_id, a, s_p, r)

                    total *= pi[state_id, a]
                    lineWorld.v[state_id] += total
                    lineWorld.v[state_id] = max(lineWorld.v[state_id], total)
                delta = max(delta, np.abs(v - lineWorld.v[state_id]))
            if delta < theta:
                break

        for s in lineWorld.states():
            best_a = -1
            best_a_score = -99999999999
            for a in lineWorld.actions():
                total = 0
                for s_p in lineWorld.states():
                    for r in range(len(lineWorld.rewards())):
                        # total += p[s, a, s_p, r] * (R[r] + 0.999 * V[s_p])
                        total += lineWorld.transition_probability(s, a, s_p, r)
                if total > best_a_score:
                    best_a = a
                    best_a_score = total
            pi[s, :] = 0.0
            pi[s, best_a] = 1.0
        return PolicyAndValueFunction(pi, lineWorld.v)


def policy_evaluation_on_grid_world(pi, gridWorld) -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    S = [x for x in range(gridWorld.nb_cells)]
    A = gridWorld.global_actions
    R = gridWorld.rewards

    theta = 0.0000001
    V = np.random.random((gridWorld.nb_cells,))
    V[gridWorld.nb_cells - 1] = 0.0  # états finaux
    V[0] = 0.0
    p = gridWorld.initialize_p(S, A, R)

    while True:
        delta = 0
        for state_id in S:
            v = V[state_id]
            V[state_id] = 0.0
            for a in A:
                total = 0.0
                for s_p in S:
                    for r in range(len(R)):
                        total += Bellman(p, state_id, a, s_p, r, R, V)

                total *= pi[state_id, a]
                V[state_id] += total
            delta = max(delta, np.abs(v - V[state_id]))
        if delta < theta:
            break
    gridWorld.view()
    return V


def policy_iteration_on_grid_world(pi, gridWorld: GridWorld) -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # TODO

    S = [x for x in range(gridWorld.nb_cells)]
    A = gridWorld.global_actions
    R = gridWorld.rewards

    theta = 0.0000001
    V = np.random.random((gridWorld.nb_cells,))
    V[gridWorld.nb_cells - 1] = 0.0  # états finaux
    V[0] = 0.0
    p = gridWorld.initialize_p(S, A, R)

    while True:
        while True:
            delta = 0
            for state_id in S:
                v = V[state_id]
                V[state_id] = 0.0
                for a in A:
                    total = 0.0
                    for s_p in S:
                        for r in range(len(R)):
                            total += Bellman(p, state_id, a, s_p, r, R, V)
                    total *= pi[state_id, a]
                    V[state_id] += total
                    V[state_id] = max(V[state_id], total)
                delta = max(delta, np.abs(v - V[state_id]))
            if delta < theta:
                break

        for s in S:
            best_a = -1
            best_a_score = -99999999999
            for a in A:
                total = 0
                for s_p in S:
                    for r in range(len(R)):
                        total += p[s, a, s_p, r] * (R[r] + 0.999 * V[s_p])
                if total > best_a_score:
                    best_a = a
                    best_a_score = total
            pi[s, :] = 0.0
            pi[s, best_a] = 1.0
        return PolicyAndValueFunction(pi, V)


def value_iteration_on_grid_world(pi, gridWorld: GridWorld, theta=0.0000001) -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    S = [x for x in range(gridWorld.nb_cells)]
    A = gridWorld.global_actions
    R = gridWorld.rewards

    V = np.random.random((gridWorld.nb_cells,))
    V[gridWorld.nb_cells - 1] = 0.0  # états finaux
    V[0] = 0.0
    p = gridWorld.initialize_p(S, A, R)

    while True:
        while True:
            delta = 0
            for state_id in S:
                v = V[state_id]
                V[state_id] = 0.0
                for a in A:
                    total = 0.0
                    for s_p in S:
                        for r in range(len(R)):
                            total += Bellman(p, state_id, a, s_p, r, R, V)
                    total *= pi[state_id, a]
                    V[state_id] += total
                    V[state_id] = max(V[state_id], total)
                delta = max(delta, np.abs(v - V[state_id]))
            if delta < theta:
                break

        for s in S:
            best_a = -1
            best_a_score = -99999999999
            for a in A:
                total = 0
                for s_p in S:
                    for r in range(len(R)):
                        total += p[s, a, s_p, r] * (R[r] + 0.999 * V[s_p])
                if total > best_a_score:
                    best_a = a
                    best_a_score = total
            pi[s, :] = 0.0
            pi[s, best_a] = 1.0
        return PolicyAndValueFunction(pi, V)


def policy_evaluation_on_secret_env1(pi) -> ValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    env = Env1()
    S = env.states()
    A = env.actions()
    R = env.rewards()

    theta = 0.000001

    V = np.random.random((len(S),))
    V[len(S) - 1] = 0

    while True:
        delta = 0
        for state in S:
            v = V[state]
            V[state] = 0
            for a in A:
                total = 0.0
                for s_p in S:
                    for r in range(len(R)):
                        total += env.transition_probability(state, a, s_p, r)
                total *= pi[state, a]
                V[state] += total
            delta = max(delta, np.abs(v - V[state]))
        if delta < theta:
            break

    return V


def policy_iteration_on_secret_env1(pi) -> PolicyAndValueFunction:
    env = Env1()
    S = env.states()
    A = env.actions()
    R = env.rewards()

    theta = 0.000001

    V = np.random.random((len(S),))
    V[len(S) - 1] = 0

    while True:
        V = policy_evaluation_on_secret_env1(pi)
        stable = True
        for state in S:
            old_pi_s = pi[state].copy()
            best_a = -1
            best_a_score = -999999999
            for a in A:
                total = 0
                for s_p in S:
                    for r in range(len(R)):
                        total += env.transition_probability(state, a, s_p, r)

                if total > best_a_score:
                    best_a = a
                    best_a_score = total

            pi[state, :] = 0.0
            pi[state, best_a] = 1.0
            if np.any(pi[state] != old_pi_s):
                stable = False
            if stable:
                return PolicyAndValueFunction(pi, V)


def value_iteration_on_secret_env1(pi) -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Prints the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    S = env.states()
    A = env.actions()
    R = env.rewards()

    states_actions = (len(S), len(env.actions()))

    theta = 0.000001

    V = np.random.random((len(S),))
    V[len(S) - 1] = 0

    while True:
        while True:
            delta = 0
            for state in S:
                v = V[state]
                V[state] = 0.0
                for a in A:
                    total = 0.0
                    for s_p in S:
                        for r in range(len(R)):
                            total += env.transition_probability(state, a, s_p, r)
                    total *= pi[state, a]
                    V[state] += total
                    V[state] = max(V[state], total)
                delta = max(delta, np.abs(v - V[state]))
            if delta < theta:
                break

        for s in S:
            best_a = -1
            best_a_score = -99999999999
            for a in A:
                total = 0
                for s_p in S:
                    for r in range(len(R)):
                        total += env.transition_probability(s, a, s_p, r)
                if total > best_a_score:
                    best_a = a
                    best_a_score = total
            pi[s, :] = 0.0
            pi[s, best_a] = 1.0
        return PolicyAndValueFunction(pi, V)


def demo():

    lineWorld = LineWorld(7)
    pi = lineWorld.create_policy("right")
    print(policy_evaluation_on_line_world(pi, lineWorld))
    print(policy_iteration_on_line_world(pi, lineWorld))
    print(value_iteration_on_line_world(pi, lineWorld))

    gridWorld = GridWorld((5, 5))
    pi = gridWorld.create_policy()
    print(policy_evaluation_on_grid_world(pi, gridWorld))
    print(policy_iteration_on_grid_world(pi, gridWorld))
    print(value_iteration_on_grid_world(pi, gridWorld))

    # Demo pour l'Env secret 1
    env = Env1()
    S = env.states()
    states_actions = (len(S), len(env.actions()))

    # pi_env1 = pi_env1 = np.ones(states_actions) * (1 / 3)
    pi_env1 = np.random.random(states_actions)
    print("pi_env1 :", pi_env1)

    print(policy_evaluation_on_secret_env1(pi_env1))
    print(policy_iteration_on_secret_env1(pi_env1))
    print(value_iteration_on_secret_env1(pi_env1))


if __name__ == '__main__':
    gridWorld = GridWorld((5, 5))
    gridWorld.view()
    exit(0)
    demo()
