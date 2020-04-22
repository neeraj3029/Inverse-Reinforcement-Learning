import numpy as np 
import matplotlib.pyplot as plt

from mdp import gridworld, value_iteration
from linear_prog_irl import *

def show_heatmap(matrix, title='', block=True, fig_num=1, text=True):
    if block:
        plt.figure(fig_num)
        plt.clf()
    plt.imshow(matrix, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    if text:
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                plt.text(x, y, '%.1f' % matrix[y, x], horizontalalignment='center', verticalalignment='center')
    if block:
        plt.ion()
        print('press enter')
        plt.show()
        input()

if __name__ == "__main__":
    height = 5
    width = 5
    N_s = height*width
    N_a = 4 # left right up and down
    R_max = 10
    trans_prob = 0.7 # with 30% prob takes random action other than chosen
    gamma = 0.9
    lmbda = 10
    iterations = 100
    grid = np.zeros((height,width))
    grid[height-1][width-1] = R_max

    gw_mdp = gridworld.GridWorld(grid, {(height-1, width-1)}, trans_prob)

    val_it = value_iteration.ValueIterationAgent(gw_mdp, gamma, iterations)

    R_mat = gw_mdp.get_reward_mat()
    # show rewards map
    show_heatmap(R_mat, 'Ground Truth of Reward')

    V_mat = gw_mdp.get_values_mat(val_it.get_values())
    # show values map
    show_heatmap(V_mat, 'Ground Truth of Value')
    P = np.zeros((N_s, N_s, N_a)) # transition matrix

    for s_i in range(N_s):
        state_i = gw_mdp.idx2pos(s_i)
        for a in range(N_a):
            probabilities = gw_mdp.get_transition_states_and_probs(state_i, a)
            for state_j, prob in probabilities:
                s_j = gw_mdp.pos2idx(state_j)
                # Prob of si to sj given action a
                P[s_i, s_j, a] = prob

    # gw_mdp.display_policy_grid(val_it.get_optimal_policy())
    # gw_mdp.display_value_grid(val_it.get_values())

    opt_policy = np.zeros(N_s)
    for i in range(N_s):
        opt_policy[i] = val_it.get_action(gw_mdp.idx2pos(i))

    # find the rewards for desired policy
    rewards = lp_irl(P, opt_policy, gamma=gamma, l1=lmbda, R_max=R_max)
    # rewards = np.zeros(N_s)
    # show new rewards
    show_heatmap(np.reshape(rewards, (height, width), order='F'), 'Reward Map - Recovered')



