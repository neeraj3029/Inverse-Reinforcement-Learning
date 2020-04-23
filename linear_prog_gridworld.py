import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from mdp import gridworld
from linear_prog_irl import *

def show_heatmap(matrix, title='', block=True, fig_num=1, text=True):
    if block:
        plt.figure(fig_num)
        plt.clf()
    plt.imshow(matrix, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.show()
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
    gamma = 0.5
    lmbda = 10
    iterations = 100
    grid = np.zeros((height,width))
    grid[height-1][width-1] = R_max

    gw_mdp = gridworld.GridWorld(grid, {(height-1, width-1)}, trans_prob)


    R_mat = gw_mdp.get_reward_mat()
    # show rewards map
    show_heatmap(R_mat, 'Ground Truth of Reward')

    P = np.zeros((N_s, N_s, N_a)) # transition matrix

    for s_i in range(N_s):
        state_i = gw_mdp.idx2pos(s_i)
        for a in range(N_a):
            probabilities = gw_mdp.get_transition_states_and_probs(state_i, a)
            for state_j, prob in probabilities:
                s_j = gw_mdp.pos2idx(state_j)
                P[s_i, s_j, a] = prob

    opt_policy = np.zeros(N_s)

    opt_policy = [2, 2, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 2, 0, 2, 2, 2, 2, 0]

    rewards = linear_prog_irl(P, opt_policy, gamma=gamma, l1=lmbda, R_max=R_max)

    # show new rewards
    rew = np.reshape(rewards, (height, width), order='F')
    print(rew)
    show_heatmap(np.reshape(rewards, (height, width), order='F'), 'Reward Map - Recovered')
    
