import numpy as np
nb_cells = 8

S = np.arange(nb_cells)
A = np.array([0, 1])  # left, Right
R = np.array([-1.0, 0.0, 1.0])


def step(s: int, a: int) -> (int, float, bool):
    if s == 0 or s == nb_cells - 1:
        return s, 0.0, True

    if a == 0:
        s_p = s - 1
        if s == 1:
            return s_p, -1.0, True
        else:
            return s_p, 0.0, False

    if a == 1:
        s_p = s + 1
        if s == nb_cells - 2:
            return s_p, 1.0, True
        else:
            return s_p, 0.0, False


def monte_carlo_control_with_exploring_starts(S, A, iter_count, max_step):
    pi = np.random.random((len(S), len(A)))
    for s in S:
        pi[s] /= np.sum(pi[s])

    q = np.random.random((len(S), len(A)))

    Returns = [[[] for a in A] for s in S]

    for it in range(iter_count):
        s0 = np.random.choice(S)
        a0 = np.random.choice(A)
        s = s0
        a = a0

        s_p, r, terminal = step(s0, a0)
        s_history = [s]
        a_history = [a]
        s_p_history = [s_p]
        r_history = [r]

        step_count = 1
        while terminal is False and step_count < max_step:
            s = s_p
            a = np.random.choice(A, p=pi[s])

            s_p, r, terminal = step(s, a)
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

            Returns[s_t][a_t].append(G)
            q[s_t, a_t] = np.mean(Returns[s_t][a_t])
            pi[s_t, :] = 0.0
            pi[s_t, np.argmax(q[s_t])] = 1.0

    return pi, q


print(monte_carlo_control_with_exploring_starts(S, A, 10000, nb_cells))

if __name__ == '__main__':
    print(monte_carlo_control_with_exploring_starts(S, A, 10000, nb_cells))
