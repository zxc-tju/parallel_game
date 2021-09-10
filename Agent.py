"""create agents for simulation"""
from gt_planner import GTPlanner
import numpy as np
import math
from scipy.optimize import minimize, fmin
from matplotlib import pyplot as plt
from tools.utility import smooth_cv

# simulation setting
dt = 0.2
TRACK_LEN = 30
MAX_DELTA_UT = 1e-4
# weights for calculate interior cost
WEIGHT_DELAY = 10
WEIGHT_DEVIATION = 0.6
WEIGHT_CHANGE = 0.2

# parameters of action bounds
MAX_STEERING_ANGLE = math.pi / 3
MAX_ACCELERATION = 3.0

# initial self IPV and guess on interacting agent's IPV
INITIAL_IPV = 0
INITIAL_IPV_GUESS = 0

# weight of interior and group cost
WEIGHT_INT = 3
WEIGHT_GRP = 1


class Agent:
    def __init__(self, position, velocity, heading, target):
        self.position = position
        self.velocity = velocity
        self.heading = heading
        self.target = target
        self.trajectory = [self.position, self.velocity, self.heading]
        self.ipv = None
        self.ipv_guess = None  # a guess on interacting counterpart's ipv
        self.trj_solution = np.repeat([position], TRACK_LEN + 1, axis=0)
        self.trj_solution_for_inter_agent = []

    def solve_game_KKT(self, inter_agent):
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

        fun = utility_KKT(self_info, inter_info)
        cons = {'type': 'ineq', 'fun': con_inter_opt(self_info, inter_info)}

        u0 = np.zeros([TRACK_LEN * 4, 1])

        bds = [(-MAX_ACCELERATION, MAX_ACCELERATION) for i in range(TRACK_LEN)] + \
              [(-MAX_STEERING_ANGLE, MAX_STEERING_ANGLE) for i in range(TRACK_LEN)]
        bds = bds + bds

        res = minimize(fun, u0, bounds=bds, method='SLSQP', constraints=cons)

        print(res.success)
        x = np.reshape(res.x, [4, TRACK_LEN]).T
        track_self = kinematic_model(x[:, 0:2], self_info[0:3])
        track_inter = kinematic_model(x[:, 2:], inter_info[0:3])
        print(x)
        return track_self, track_inter

    def solve_game_IBR(self, inter_track):
        self_info = [self.position,
                     self.velocity,
                     self.heading,
                     self.ipv,
                     self.target]
        fun = utility_IBR(self_info, inter_track)
        u0 = np.zeros([TRACK_LEN * 2, 1])
        bds = [(-MAX_ACCELERATION, MAX_ACCELERATION) for i in range(TRACK_LEN)] + \
              [(-MAX_STEERING_ANGLE, MAX_STEERING_ANGLE) for i in range(TRACK_LEN)]

        res = minimize(fun, u0, bounds=bds, method='SLSQP')

        print(res.success)
        x = np.reshape(res.x, [2, TRACK_LEN]).T
        track = kinematic_model(x, self_info[0:3])
        print(x)
        return track


def utility_KKT(self_info, inter_info):
    def fun(u):
        """
        Calculate the utility from the perspective of "self" agent

        :param u: num_step by 4 array, where the 1\2 columns are control of "self" agent
                  and 3\4 columns "inter" agent
        :return: utility of the "self" agent under the control u
        """
        track_self = kinematic_model(u[0:TRACK_LEN * 2], self_info[0:3])
        track_inter = kinematic_model(u[TRACK_LEN * 2:], inter_info[0:3])
        track_all = [track_self[:, 0:2], track_inter[:, 0:2]]
        # print(np.sin(self_info[3]))
        ut_self = np.cos(self_info[3]) * cal_interior_cost(track_self, self_info[4]) + \
                  np.sin(self_info[3]) * cal_group_cost(track_all)

        ut_inter = np.cos(inter_info[3]) * cal_interior_cost(track_inter, inter_info[4]) + \
                   np.sin(inter_info[3]) * cal_group_cost(track_all)
        return ut_self + ut_inter
    return fun


def utility_IBR(self_info, inter_track):
    def fun(u):
        """
        Calculate the utility from the perspective of "self" agent

        :param u: num_step by 4 array, where the 1\2 columns are control of "self" agent
                  and 3\4 columns "inter" agent
        :return: utility of the "self" agent under the control u
        """
        track_self = kinematic_model(u, self_info[0:3])
        track_inter = inter_track
        track_all = [track_self[:, 0:2], track_inter[:, 0:2]]
        # print(np.sin(self_info[3]))
        ut = np.cos(self_info[3]) * cal_interior_cost(track_self, self_info[4]) + \
             np.sin(self_info[3]) * cal_group_cost(track_all)
        return ut
    return fun


def cal_interior_cost(track, target):
    cv, s = get_central_vertices(target)

    # find the on-reference point of the track starting
    init_dis2cv = np.linalg.norm(cv - track[0, 0:2], axis=1)
    init_min_dis2cv = np.amin(init_dis2cv)
    init_index = np.where(init_min_dis2cv == init_dis2cv)

    # find the on-reference point of the track end
    end_dis2cv = np.linalg.norm(cv - track[-1, 0:2], axis=1)
    end_init_dis2cv = np.amin(end_dis2cv)
    end_index = np.where(end_init_dis2cv == end_dis2cv)

    # calculate the on-reference distance of the given track (the longer the better)
    travel_distance = s[end_index] - s[init_index]
    # 1. cost of travel delay
    # cost_travel_distance = - travel_distance / (np.size(track, 0) - 1)
    cost_travel_distance = - travel_distance
    # print('cost of travel delay:', cost_travel_distance)
    # initialize an array to store distance from each point in the track to cv
    dis2cv = np.zeros([np.size(track, 0), 1])
    for i in range(np.size(track, 0)):
        dis2cv[i] = np.amin(np.linalg.norm(cv - track[i, 0:2], axis=1))

    # 2. cost of lane deviation
    cost_mean_deviation = dis2cv.mean()
    # print('cost of lane deviation:', cost_mean_deviation)

    # 3. cost of change plan
    cost_plan_change = 0

    # overall cost
    cost_interior = WEIGHT_DELAY * cost_travel_distance + \
                    WEIGHT_DEVIATION * cost_mean_deviation + \
                    WEIGHT_CHANGE * cost_plan_change
    # print('interior cost:', cost_interior)
    return cost_interior * WEIGHT_INT


def cal_group_cost(track_packed):
    track_self, track_inter = track_packed
    rel_distance = np.linalg.norm(track_self - track_inter, axis=1)
    min_rel_distance = np.amin(rel_distance)
    min_index = np.where(min_rel_distance == rel_distance)[0]
    cost_group = -min_rel_distance * min_index[0] / (np.size(track_self, 0))

    # print('group cost:', cost_group)
    return cost_group * WEIGHT_GRP


def con_inter_opt(self_info, inter_info):
    def con(u):
        track_self = kinematic_model(u[0:TRACK_LEN * 2], self_info[0:3])
        # track_inter = kinematic_model(u[TRACK_LEN * 2:], inter_info[0:3])
        track_inter1 = kinematic_model(1.01 * u[TRACK_LEN * 2:], inter_info[0:3])
        track_inter2 = kinematic_model(0.99 * u[TRACK_LEN * 2:], inter_info[0:3])

        ut_inter1 = np.cos(inter_info[3]) * cal_interior_cost(track_inter1, inter_info[4]) + \
                    np.sin(inter_info[3]) * cal_group_cost([track_self, track_inter1])
        ut_inter2 = np.cos(inter_info[3]) * cal_interior_cost(track_inter2, inter_info[4]) + \
                    np.sin(inter_info[3]) * cal_group_cost([track_self, track_inter2])

        delta_ut = np.abs(ut_inter2 - ut_inter1)
        print('delta_ut', MAX_DELTA_UT - delta_ut[0])

        return MAX_DELTA_UT - delta_ut[0]
        # return delta_ut

    return con


def get_central_vertices(cv_type):
    cv_init = None
    if cv_type == 'lt':  # left turn
        cv_init = np.array([[0, -15], [5, -14.14], [10.6, -10.6], [15, 0], [15, 100]])
    elif cv_type == 'gs':  # go straight
        cv_init = np.array([[20, -2], [10, -2], [0, -2], [-150, -2]])
    assert cv_init is not None
    cv_smoothed, s_accumulated = smooth_cv(cv_init)
    return cv_smoothed, s_accumulated


def kinematic_model(u, init_state):
    if not np.size(u, 0) == TRACK_LEN:
        u = np.array([u[0:TRACK_LEN], u[TRACK_LEN:]]).T
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


if __name__ == '__main__':
    "test kinematic_model"
    # action = np.concatenate((1 * np.ones((TRACK_LEN, 1)),  0.3 * np.ones((TRACK_LEN, 1))), axis=1)
    # position = np.array([0, 0])
    # velocity = np.array([3, 3])
    # heading = np.array([math.pi/4])
    # init_info = [position, velocity, heading]
    # trajectory = kinematic_model(action, init_info)
    # x = trajectory[:, 0]
    # y = trajectory[:, 1]
    # plt.figure()
    # plt.plot(x, y, 'r-')
    # # plt.plot(trajectory[:, 3], 'r-')
    # plt.axis('equal')
    # plt.show()

    "test get_central_vertices"
    # target = 'gs'
    # cv, s = get_central_vertices(target)
    # x = cv[:, 0]
    # y = cv[:, 1]
    # plt.plot(x, y, 'r-')
    # plt.axis('equal')
    # plt.show()

    "test cal_cost"
    # track_test = [np.array([[0, -15], [5, -13], [10, -10]]), np.array([[20, -2], [10, -2], [0, -2]])]
    # # track_test = [np.array([[0, -15], [5, -13], [10, -10]]), np.array([[0, -14], [5, -11], [10, -9]])]
    # target1 = 'lt'
    # target2 = 'gs'
    # cost_it = cal_group_cost(track_test)
    # print('cost is :', cost_it)

    "test solve game"
    init_position_lt = np.array([13, -7])
    init_velocity_lt = np.array([0, 2])
    init_heading_lt = np.array([math.pi / 3])
    # initial state of the go-straight vehicle
    init_position_gs = np.array([20, -2])
    init_velocity_gs = np.array([-2, 0])
    init_heading_gs = np.array([math.pi])

    # generate LT and GS agents
    agent_lt = Agent(init_position_lt, init_velocity_lt, init_heading_lt, 'lt')
    agent_lt.ipv = INITIAL_IPV
    agent_lt.ipv_guess = INITIAL_IPV_GUESS
    agent_gs = Agent(init_position_gs, init_velocity_gs, init_heading_gs, 'gs')
    agent_gs.ipv = INITIAL_IPV
    agent_gs.ipv_guess = INITIAL_IPV_GUESS

    # track_s, track_i = agent_lt.solve_game_KKT(agent_gs)

    track_s = agent_lt.solve_game_IBR(agent_gs.trj_solution)
    agent_lt.trj_solution = track_s
    track_i = agent_gs.solve_game_IBR(agent_lt.trj_solution)
    agent_gs.trj_solution = track_i

    cv_init_it, _ = get_central_vertices('lt')
    cv_init_gs, _ = get_central_vertices('gs')
    plt.plot(cv_init_gs[:, 0], cv_init_gs[:, 1], 'b--')
    for t in range(TRACK_LEN):
        plt.plot(cv_init_it[:, 0], cv_init_it[:, 1], 'r-')
        plt.plot(cv_init_gs[:, 0], cv_init_gs[:, 1], 'b-')
        plt.plot(track_s[t, 0], track_s[t, 1], 'r*')
        plt.plot(track_i[t, 0], track_i[t, 1], 'b*')
        plt.axis('equal')
        plt.xlim(5, 25)
        plt.ylim(-10, 10)
        plt.pause(0.1)
    plt.show()
