import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_match3.envs.game import Game, Point
from gym_match3.envs.game import OutOfBoardError, ImmovableShapeError
from gym_match3.envs.levels import LEVELS, Match3Levels
from gym_match3.envs.renderer import Renderer

import numpy as np

from itertools import product
import warnings

BOARD_NDIM = 2


class Match3Env(gym.Env):
    metadata = {'render.modes': None}

    def __init__(self, rollout_len=100, all_moves=False, levels=None, random_state=None):
        self.rollout_len = rollout_len
        self.random_state = random_state
        self.all_moves = all_moves
        self.levels = levels or Match3Levels(LEVELS)
        self.h = self.levels.h
        self.w = self.levels.w
        self.n_shapes = self.levels.n_shapes
        self.__episode_counter = 0

        self.__game = Game(
            rows=self.h,
            columns=self.w,
            n_shapes=self.n_shapes,
            length=3,
            all_moves=all_moves,
            random_state=self.random_state)
        self.reset()
        self.renderer = Renderer(self.n_shapes)

        # setting observation space
        self.observation_space = spaces.Box(
            low=0,
            high=self.n_shapes,
            shape=(self.h, self.w, self.n_shapes,),
            dtype=int)

        # setting actions space
        self.__match3_actions = self.__get_available_actions()
        self.action_space = spaces.Discrete(
            len(self.__match3_actions))

    @staticmethod
    def __get_directions(board_ndim):
        """ get available directions for any number of dimensions """
        directions = [
            [[0 for _ in range(board_ndim)] for _ in range(2)]
            for _ in range(board_ndim)
        ]
        for ind in range(board_ndim):
            directions[ind][0][ind] = 1
            directions[ind][1][ind] = -1
        return directions

    def __points_generator(self):
        """ iterates over points on the board """
        rows, cols = self.__game.board.board_size
        points = [Point(i, j) for i, j in product(range(rows), range(cols))]
        for point in points:
            yield point

    def __get_available_actions(self):
        """ calculate available actions for current board sizes """
        # actions = []
        # directions = self.__get_directions(board_ndim=BOARD_NDIM)
        # for point in self.__points_generator():
        #     for axis_dirs in directions:
        #         for dir_ in axis_dirs:
        #             dir_p = Point(*dir_)
        #             new_point = point + dir_p
        #             try:
        #                 _ = self.__game.board[new_point]
        #                 if not ((point, new_point) in actions or (new_point, point) in actions):
        #                     actions.append((point, new_point))
        #             except OutOfBoardError:
        #                 continue
        # actions = list(dict.fromkeys(actions))
        # print(actions)
        """
        Enumerate the actions by hand (note that this wont work if we have non-allowed moves).
        It first goes row by row (horizontal actions) and then vertical actions.
        """
        actions = []

        rows, cols = self.__game.board.board_size

        for i in range(rows):
            for j in range(cols-1):
                actions.append((Point(i,j), Point(i, j+1)))

        for j in range(cols):
            for i in range(rows-1):
                actions.append((Point(i,j), Point(i+1, j)))
   
        return actions

    def __get_action(self, ind):
        return self.__match3_actions[ind]

    def step(self, action):
        # make action
        m3_action = self.__get_action(action)
        reward = self.__swap(*m3_action)

        # change counter even action wasn't successful
        self.__episode_counter += 1
        if self.__episode_counter >= self.rollout_len:
            episode_over = True
            self.__episode_counter = 0
            ob = self.reset()
        else:
            episode_over = False
            ob = self.__binarize_state(self.__get_board())
        return ob, reward, episode_over, {}

    def reset(self, *args, **kwargs):
        board = self.levels.sample()
        self.__game.start(board)
        return   self.__binarize_state(self.__get_board())

    def __swap(self, point1, point2):
        possible_moves = self.__game._Game__get_possible_moves()
        try:
            score = self.__game.swap(self.__game.board, point1, point2)
            reward = [score[0], score[1], possible_moves]
            # tiles_matched = score[0] if isinstance(score[0], (int, float)) else 0
            # if tiles_matched > 0:
                # reward = [tiles_matched, tiles_matched, possible_moves]
            # else:
                # reward = [0, -1, possible_moves]
        except ImmovableShapeError:
            reward = [0, -100, possible_moves]
        return reward
    
    def __get_legal_actions(self):
        # Return a list of boolean corresponding to the legal actions
        
        print(dir(Game))
        legal_moves = self.__game._Game__get_possible_moves()
        
        # Formatting possible moves correctly
        for i in range(len(possible_moves)):
            print(i)
            print(self.rows - possible_moves[i][0].get_coord()[0])
            legal_moves[i] = list(legal_moves[i])
            legal_moves[i][0].set_coord_row(self.rows - possible_moves[i][0].get_coord()[0])
            legal_moves[i][0].set_coord_col(possible_moves[i][0].get_coord()[1])
            legal_moves[i][1] = Point(legal_moves[i][0].get_coord()[0] - possible_moves[i][1][0], possible_moves[i][1][1])
            print(legal_moves[i])

        # Get actions and compare to possible moves
        legal_moves_bool = []
        available_actions = self.__get_available_actions()

        for i in range(len(possible_moves)):
            if(legal_moves[i] == available_actions[i]):
                legal_moves_bool.append(True)
            else:
                legal_moves_bool.append(False)

        print(legal_moves_bool)
        return legal_moves_bool

    def __get_board(self):
        return self.__game.board.board.copy()

    def render(self, mode='human', close=False):
        if close:
            warnings.warn("close=True isn't supported yet")

        import matplotlib.pyplot as plt
        _, ax = plt.subplots(1, 1)
        self.renderer.render_board(self.__game.board, ax)
        plt.show()
        
    def __binarize_state(self, state):
        binary_mat = np.zeros((state.shape[0], state.shape[1], self.n_shapes))
        for gem in range(self.n_shapes):
            binary_mat[:, :, gem] = (state == gem).astype(int)
        return binary_mat
