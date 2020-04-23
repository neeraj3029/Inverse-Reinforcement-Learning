# Gridworld provides a basic environment for RL agents to interact with
#
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
#
# MIT License

import numpy as np


class GridWorld(object):
  """
  Grid world environment
  """

  def __init__(self, grid, terminals, trans_prob=1):
    """
    input:
      grid        2-d list of the grid including the reward
      terminals   a set of all the terminal states
      trans_prob  transition probability when given a certain action
    """
    self.height = len(grid)
    self.width = len(grid[0])
    self.n_states = self.height*self.width
    for i in range(self.height):
      for j in range(self.width):
        grid[i][j] = str(grid[i][j])


    self.terminals = terminals
    self.grid = grid
    self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    self.actions = [0, 1, 2, 3]
    self.n_actions = len(self.actions)
    # self.dirs = {0: 's', 1: 'r', 2: 'l', 3: 'd', 4: 'u'}
    self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u'}
    #              right,    left,   down,   up 
    # self.action_nei = {0: (0,1), 1:(0,-1), 2:(1,0), 3:(-1,0)}

    # If the mdp is deterministic, the transition probability of taken a certain action should be 1
    # otherwise < 1, the rest of the probability are equally spreaded onto
    # other neighboring states.
    self.trans_prob = trans_prob



  def get_transition_states_and_probs(self, state, action):
    """
    get all the possible transition states and their probabilities with [action] on [state]
    args
      state     (y, x)
      action    int
    returns
      a list of (state, probability) pair
    """
    if self.is_terminal(tuple(state)):
      return [(tuple(state), 1)]

    if self.trans_prob == 1:
      inc = self.neighbors[action]
      nei_s = [state[0] + inc[0], state[1] + inc[1]]
      if nei_s[0] >= 0 and nei_s[0] < self.height and nei_s[1] >= 0 and nei_s[1] < self.width and self.grid[int(nei_s[0])][int(nei_s[1])] != 'x':
        return [(nei_s, 1)]
      else:
        # if the state is invalid, stay in the current state
        return [(state, 1)]
    else:
      # [(0, 1), (0, -1), (1, 0), (-1, 0)]
      mov_probs = np.zeros([self.n_actions])
      mov_probs[action] = self.trans_prob
      mov_probs += (1-self.trans_prob)/self.n_actions

      for a in range(self.n_actions):
        inc = self.neighbors[a]
        nei_s = (state[0] + inc[0], state[1] + inc[1])
        if nei_s[0] < 0 or nei_s[0] >= self.height or nei_s[1] < 0 or nei_s[1] >= self.width or self.grid[int(nei_s[0])][int(nei_s[1])] == 'x':
          # if the move is invalid, accumulates the prob to the current state
          mov_probs[self.n_actions-1] += mov_probs[a]
          mov_probs[a] = 0

      res = []
      for a in range(self.n_actions):
        if mov_probs[a] != 0:
          inc = self.neighbors[a]
          nei_s = (state[0] + inc[0], state[1] + inc[1])
          res.append((nei_s, mov_probs[a]))
      return res


  def is_terminal(self, state):
    """
    returns
      True if the [state] is terminal
    """
    if tuple(state) in self.terminals:
      return True
    else:
      return False

  def get_reward_mat(self):
    """
    Get reward matrix from gridworld
    """
    shape = np.shape(self.grid)
    r_mat = np.zeros(shape)
    for i in range(shape[0]):
      for j in range(shape[1]):
        r_mat[i, j] = float(self.grid[i][j])
    return r_mat

  def pos2idx(self, pos):
    """
    input:
      column-major 2d position
    returns:
      1d index
    """
    return pos[0] + pos[1] * self.height

  def idx2pos(self, idx):
    """
    input:
      1d idx
    returns:
      2d column-major position
    """
    return (idx % self.height, idx // self.height)
