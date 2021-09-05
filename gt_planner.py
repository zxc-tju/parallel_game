import numpy as np
from scipy.optimize import minimize

TIME_HORIZON = 10
dt = 0.2


class GTPlanner:
    def __init__(self, sub_info, inter_info):
        self.player1 = Player(sub_info)
        self.player2 = Player(inter_info)

    # def gt_planning(self):

        # if self.player1.target == 'turn-left':
        #     utility = utility_lt(self.player1.ipv, self.player2.ipv)


class Player:
    def __init__(self, info):
        self.position = info[0]
        self.velocity = info[1]
        self.heading = info[2]
        self.ipv = info[3]
        self.target = info[4]
        self.trajectory = []

    # def initialize(self):
    #     if self.target == 'turn-left':
    #         utility = utility_lt(self.ipv, [])
    #     u_init = [np.ones(np.floor(TIME_HORIZON/dt), 1),np.ones(np.floor(TIME_HORIZON/dt), 1)]
    #
    # def motion_planning(self, obstacle_traj):
    #     """
    #
    #     :param obstacle_traj: trajectory of an obstacle
    #     :return:
    #     """
    #     if self.target == 'turn-left':
    #         utility = utility_lt(self.ipv, obstacle_traj)


# def utility_lt(ipv1, ipv2):
#     def fun(x):
#         cost1 = np.cos(ipv1)*cal_cost(x)
#         return np.cos(ipv1)*np.sum(x)
#     return fun
#
# def cal_cost():
#
# def constrain(ub,lb):
#     def con(x):
#         return np.abs(x[])
#     return fun