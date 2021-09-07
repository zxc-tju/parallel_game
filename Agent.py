"""create agents for simulation"""
from gt_planner import GTPlanner
import numpy as np
import math
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from tools.utility import smooth_cv

dt = 0.2
# weights for calculate interior cost
WEIGHT_DELAY = 1
WEIGHT_DEVIATION = 0.5
WEIGHT_CHANGE = 0.2


class Agent:
    def __init__(self, position, velocity, heading, target):
        self.position = position
        self.velocity = velocity
        self.heading = heading
        self.target = target
        self.trajectory = [self.position, self.velocity, self.heading]
        self.ipv = None
        self.ipv_guess = None  # a guess on interacting counterpart's ipv
        self.trj_solution = []
        self.trj_solution_for_inter_agent = []

    def solve_game(self, inter_agent):
        """
        solve the game with an interacting counterpart
        :param inter_agent: interacting counterpart [Agent]
        :return:
        """
        # accessible info. on interacting counterpart:
        #   1. trajectory history: inter_agent.trajectory
        #   2. ipv guess: self.ipv_guess
        self_info = [self.position,
                     self.velocity,
                     self.heading,
                     self.ipv,
                     self.target]
        inter_info = [inter_agent.position,
                      inter_agent.velocity,
                      inter_agent.heading,
                      self.ipv_guess,
                      inter_agent.target]

        fun = utility_lt(self_info, inter_info)


def utility_lt(self_info, inter_info):
    def fun(u):
        """
        Calculate the utility from the perspective of "self" agent

        :param u: num_step by 4 array, where the 1\2 columns are control of "self" agent
                  and 3\4 columns "inter" agent
        :return: utility of the "self" agent under the control x
        """
        track_self = kinematic_model(u[:, 0:1], self_info[0:2])
        track_inter = kinematic_model(u[:, 0:1], self_info[0:2])
        track_all = [track_self, track_inter]
        ut_self = np.cos(self_info[3]) * cal_cost_interior(track_self, self_info[4]) + \
                  np.sin(self_info[3]) * cal_cost_group(track_all, self_info, inter_info[4])

        ut_inter = np.cos(inter_info[3]) * cal_cost_interior(track_inter, inter_info[4]) + \
                   np.sin(inter_info[3]) * cal_cost_group(track_all, self_info, inter_info[4])

        return ut_self + ut_inter

    return fun


def kinematic_model(u, init_state):
    r_len = 0.8
    f_len = 1
    pos, v, h = init_state
    x = pos[0]
    y = pos[1]
    psi = h[0]
    v = np.sqrt(v[0] ** 2 + v[1] ** 2)

    track = [[x, y, psi, v]]
    for i in range(len(u)):
        a = u[i][0]
        delta = u[i][1]
        beta = math.atan((r_len / (r_len + f_len)) * math.tan(delta))
        x = x + v * np.cos(psi + beta) * dt
        y = y + v * np.sin(psi + beta) * dt
        psi = psi + (v / f_len) * np.sin(beta) * dt
        v = v + a * dt
        track.append([x, y, psi, v])
    return np.array(track)


def cal_cost_interior(track, target):
    cv, s = get_central_vertices(target)

    # find the on-reference point of the track starting
    # test = cv - track[0, ]
    init_dis2cv = np.linalg.norm(cv - track[0, ], axis=1)
    init_min_dis2cv = np.amin(init_dis2cv)
    init_index = np.where(init_min_dis2cv == init_dis2cv)

    # find the on-reference point of the track end
    end_dis2cv = np.linalg.norm(cv - track[-1, ], axis=1)
    end_init_dis2cv = np.amin(end_dis2cv)
    end_index = np.where(end_init_dis2cv == end_dis2cv)

    # calculate the on-reference distance of the given track (the longer the better)
    travel_distance = s[end_index] - s[init_index]
    # 1. cost of travel delay
    cost_travel_distance = - travel_distance / (np.size(track, 0) - 1)
    print('cost of travel delay:',  cost_travel_distance)
    # initialize an array to store distance from each point in the track to cv
    dis2cv = np.zeros([np.size(track, 0), 1])
    for i in range(np.size(track, 0)):
        dis2cv[i] = np.amin(np.linalg.norm(cv - track[i, ], axis=1))

    # 2. cost of lane deviation
    cost_mean_deviation = dis2cv.mean()
    print('cost of lane deviation:', cost_mean_deviation)

    # 3. cost of change plan
    cost_plan_change = 0

    # overall cost
    cost_interior = WEIGHT_DELAY * cost_travel_distance + \
                    WEIGHT_DEVIATION * cost_mean_deviation + \
                    WEIGHT_CHANGE * cost_plan_change

    return cost_interior


def cal_cost_group(x_bi, target1, target2):
    cost_group = 0
    return cost_group


def get_central_vertices(cv_type):
    cv_init = None
    if cv_type == 'lt':  # left turn
        cv_init = np.array([[0, -15], [5, -14.14], [10.6, -10.6], [15, 0]])
    elif cv_type == 'gs':  # go straight
        cv_init = np.array([[20, -2], [10, -2], [0, -2], [-20, -2]])
    assert cv_init is not None
    cv_smoothed, s_accumulated = smooth_cv(cv_init)
    return cv_smoothed, s_accumulated


if __name__ == '__main__':
    "test kinematic_model"
    # action = np.concatenate((0 * np.ones((100, 1)),  0.5 * np.ones((100, 1))), axis=1)
    # position = np.array([0, 0])
    # velocity = np.array([3, 3])
    # heading = np.array([math.pi/4])
    # init_info = [position, velocity, heading]
    # trajectory = kinematic_model(action, init_info)
    # x = trajectory[:, 0]
    # y = trajectory[:, 1]
    # plt.figure()
    # plt.plot(x, y, 'r-')
    # plt.axis('equal')
    # plt.show()

    # "test get_central_vertices"
    # target = 'gs'
    # cv, s = get_central_vertices(target)
    # x = cv[:, 0]
    # y = cv[:, 1]
    # plt.plot(x, y, 'r-')
    # plt.axis('equal')
    # plt.show()

    "test cal_cost_interior"
    track_test = np.array([[0, -15], [5, -13], [10, -10]])
    target = 'lt'
    cost_it = cal_cost_interior(track_test, target)
    print('cost is :', cost_it)
