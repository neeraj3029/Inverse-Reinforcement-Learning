import numpy as np
import scipy.optimize as opt

def normalize(val):
    min_val = np.min(val)
    max_val = np.max(val)

    return (val - min_val) / (max_val - min_val)

def linear_prog_irl(trans_probs, policy, gamma=0.1, l1=0.5, R_max=10):
    
    N_s, _, N_a = np.shape(trans_probs)
    N_s = int(N_s)
    N_a = int(N_a)

    # Formulate a linear IRL problem
    A = np.zeros([2 * N_s * (N_a + 1), 3 * N_s])
    b = np.zeros([2 * N_s * (N_a + 1)])
    c = np.zeros([3 * N_s])

    for i in range(N_s):
        a_opt = int(policy[i])
        tmp_inv = np.linalg.inv(np.identity(N_s) - gamma * trans_probs[:, :, a_opt])

        cnt = 0
        for a in range(N_a):
            if a != a_opt:
                A[i * (N_a - 1) + cnt, :N_s] = - \
                    np.dot(trans_probs[i, :, a_opt] - trans_probs[i, :, a], tmp_inv)
                A[N_s * (N_a - 1) + i * (N_a - 1) + cnt, :N_s] = - \
                    np.dot(trans_probs[i, :, a_opt] - trans_probs[i, :, a], tmp_inv)
                A[N_s * (N_a - 1) + i * (N_a - 1) + cnt, N_s + i] = 1
                cnt += 1

    for i in range(N_s):
        A[2 * N_s * (N_a - 1) + i, i] = 1
        b[2 * N_s * (N_a - 1) + i] = R_max

    for i in range(N_s):
        A[2 * N_s * (N_a - 1) + N_s + i, i] = -1
        b[2 * N_s * (N_a - 1) + N_s + i] = 0

    for i in range(N_s):
        A[2 * N_s * (N_a - 1) + 2 * N_s + i, i] = 1
        A[2 * N_s * (N_a - 1) + 2 * N_s + i, 2 * N_s + i] = -1

    for i in range(N_s):
        A[2 * N_s * (N_a - 1) + 3 * N_s + i, i] = 1
        A[2 * N_s * (N_a - 1) + 3 * N_s + i, 2 * N_s + i] = -1

    for i in range(N_s):
        c[N_s:2 * N_s] = -1
        c[2 * N_s:] = l1

    res = opt.linprog(c, A_ub=A, b_ub=b)
    reward = res.x[:N_s]
    reward = normalize(reward) * R_max

    return reward
        
    


