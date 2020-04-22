import numpy as np
import scipy.optimize as opt

def normalize(val):
    min_va

def lp_irl(trans_probs, policy, gamma = 0.1, l1 = 0.5, R_max = 10):
    
    N_STATES, _, N_ACTIONS = np.shape(trans_probs)
    N_STATES = int(N_STATES)
    N_ACTIONS = int(N_ACTIONS)

    # Formulate a linear IRL problem
    A = np.zeros([2 * N_STATES * (N_ACTIONS + 1), 3 * N_STATES])
    b = np.zeros([2 * N_STATES * (N_ACTIONS + 1)])
    c = np.zeros([3 * N_STATES])

    for i in range(N_STATES):
        a_opt = int(policy[i])
        tmp_inv = np.linalg.inv(np.identity(N_STATES) - gamma * trans_probs[:, :, a_opt])

        cnt = 0
        for a in range(N_ACTIONS):
            if a != a_opt:
                A[i * (N_ACTIONS - 1) + cnt, :N_STATES] = - \
                    np.dot(trans_probs[i, :, a_opt] - trans_probs[i, :, a], tmp_inv)
                A[N_STATES * (N_ACTIONS - 1) + i * (N_ACTIONS - 1) + cnt, :N_STATES] = - \
                    np.dot(trans_probs[i, :, a_opt] - trans_probs[i, :, a], tmp_inv)
                A[N_STATES * (N_ACTIONS - 1) + i * (N_ACTIONS - 1) + cnt, N_STATES + i] = 1
                cnt += 1

    for i in range(N_STATES):
        A[2 * N_STATES * (N_ACTIONS - 1) + i, i] = 1
        b[2 * N_STATES * (N_ACTIONS - 1) + i] = R_max

    for i in range(N_STATES):
        A[2 * N_STATES * (N_ACTIONS - 1) + N_STATES + i, i] = -1
        b[2 * N_STATES * (N_ACTIONS - 1) + N_STATES + i] = 0

    for i in range(N_STATES):
        A[2 * N_STATES * (N_ACTIONS - 1) + 2 * N_STATES + i, i] = 1
        A[2 * N_STATES * (N_ACTIONS - 1) + 2 * N_STATES + i, 2 * N_STATES + i] = -1

    for i in range(N_STATES):
        A[2 * N_STATES * (N_ACTIONS - 1) + 3 * N_STATES + i, i] = 1
        A[2 * N_STATES * (N_ACTIONS - 1) + 3 * N_STATES + i, 2 * N_STATES + i] = -1

    for i in range(N_STATES):
        c[N_STATES:2 * N_STATES] = -1
        c[2 * N_STATES:] = l1

    res = opt.linprog(c, A_ub=A, b_ub=b)
    reward = res.x[:N_STATES]

    return reward
        
    


