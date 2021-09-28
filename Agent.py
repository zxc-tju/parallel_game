"""create agents for simulation"""
import numpy as np
import math
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from tools.utility import get_central_vertices, kinematic_model
import copy

# simulation setting
dt = 0.1
TRACK_LEN = 10
MAX_DELTA_UT = 1e-4
# weights for calculate interior cost
WEIGHT_DELAY = 2
WEIGHT_DEVIATION = 0.4
WEIGHT_CHANGE = 0.2

# parameters of action bounds
MAX_STEERING_ANGLE = math.pi / 6
MAX_ACCELERATION = 3.0

# initial guess on interacting agent's IPV
INITIAL_IPV_GUESS = 0

# weight of interior and group cost
WEIGHT_INT = 5
WEIGHT_GRP = 1.5

# likelihood function
sigma = math.pi / 40


class Agent:
    def __init__(self, position, velocity, heading, target):
        self.position = position
        self.velocity = velocity
        self.heading = heading
        self.target = target
        # conducted trajectory
        self.trajectory = np.array([[self.position[0],
                                    self.position[1],
                                    self.velocity[0],
                                    self.velocity[1],
                                    self.heading], ])
        # trajectory plan at each time step
        self.trj_solution = np.repeat([position], TRACK_LEN + 1, axis=0)
        # collection of trajectory plans at every time step
        self.trj_solution_collection = []
        self.estimated_inter_agent = None
        self.ipv = 0
        self.ipv_collection = []
        self.ipv_error_collection = []
        self.virtual_track_collection = []

    # def solve_game_KKT(self, inter_agent):
    #     """
    #     solve the game with an interacting counterpart
    #     :param inter_agent: interacting counterpart [Agent]
    #     :return:
    #     """
    #     # accessible info. on interacting counterpart:
    #     #   1. trajectory history: inter_agent.trajectory
    #     #   2. ipv guess: self.ipv_guess
    #     self_info = [self.position,
    #                  self.velocity,
    #                  self.heading,
    #                  self.ipv,
    #                  self.target]
    #     inter_info = [inter_agent.position,
    #                   inter_agent.velocity,
    #                   inter_agent.heading,
    #                   self.ipv_guess,
    #                   inter_agent.target]
    #
    #     fun = utility_KKT(self_info, inter_info)
    #     cons = {'type': 'ineq', 'fun': con_inter_opt(self_info, inter_info)}
    #
    #     u0 = np.zeros([TRACK_LEN * 4])
    #
    #     bds = [(-MAX_ACCELERATION, MAX_ACCELERATION) for i in range(TRACK_LEN)] + \
    #           [(-MAX_STEERING_ANGLE, MAX_STEERING_ANGLE) for i in range(TRACK_LEN)]
    #     bds = bds + bds
    #
    #     res = minimize(fun, u0, bounds=bds, method='trust-constr', constraints=cons, options={'disp': True})
    #     print(res.success)
    #     x = np.reshape(res.x, [4, TRACK_LEN]).T
    #     track_self = kinematic_model(x[:, 0:2], self_info[0:3])
    #     track_inter = kinematic_model(x[:, 2:], inter_info[0:3])
    #     print(x)
    #     return track_self, track_inter

    def solve_game_IBR(self, inter_track):
        self_info = [self.position,
                     self.velocity,
                     self.heading,
                     self.ipv,
                     self.target]
        p, v, h = self_info[0:3]
        init_state_4_kine = [p[0], p[1], v[0], v[1], h]
        fun = utility_IBR(self_info, inter_track)
        u0 = np.zeros([TRACK_LEN * 2, 1])
        bds = [(-MAX_ACCELERATION, MAX_ACCELERATION) for i in range(TRACK_LEN)] + \
              [(-MAX_STEERING_ANGLE, MAX_STEERING_ANGLE) for i in range(TRACK_LEN)]

        res = minimize(fun, u0, bounds=bds, method='SLSQP')

        # print(res.success)
        x = np.reshape(res.x, [2, TRACK_LEN]).T
        self.trj_solution = kinematic_model(x, init_state_4_kine, TRACK_LEN, dt)

    def update_state(self, inter_agent, virtual_agent_IPV_range):
        self.position = self.trj_solution[1, 0:2]
        self.velocity = self.trj_solution[1, 2:4]
        self.heading = self.trj_solution[1, -1]
        self.estimated_inter_agent.position = inter_agent.trj_solution[1, 0:2]
        self.estimated_inter_agent.velocity = inter_agent.trj_solution[1, 2:4]
        self.estimated_inter_agent.heading = inter_agent.trj_solution[1, -1]

        new_track_point = np.array([[self.position[0],
                                    self.position[1],
                                    self.velocity[0],
                                    self.velocity[1],
                                    self.heading], ])
        self.trajectory = np.concatenate((self.trajectory, new_track_point), axis=0)

        self.trj_solution_collection.append(self.trj_solution)

        # update IPV
        current_time = np.size(self.trajectory, 0) - 1
        if current_time > 1:
            start_time = max(0, current_time - 6)
            time_duration = current_time - start_time

            candidates = self.estimated_inter_agent.virtual_track_collection[start_time]
            var = np.zeros(len(candidates))

            for i in range(len(candidates)):
                virtual_track = candidates[i][0:time_duration, 0:2]
                actual_track = inter_agent.trajectory[start_time:current_time, 0:2]
                rel_dis = np.linalg.norm(virtual_track-actual_track, axis=0)
                # likelihood of each candidate
                var[i] = np.log10(np.prod((1/sigma/np.sqrt(2*math.pi))*np.exp(-rel_dis**2/(2*sigma**2))))
                if var[i] < 0:
                    var[i] = 0

            weight = var/(sum(var) + 1e-6)
            print(weight)

            # weighted sum of all candidates' IPVs
            self.estimated_inter_agent.ipv = sum(virtual_agent_IPV_range * weight)

            # save updated ipv and estimation error
            self.estimated_inter_agent.ipv_collection.append(self.estimated_inter_agent.ipv)
            error = 1 - np.sqrt(sum(weight ** 2))
            self.estimated_inter_agent.ipv_error_collection.append(error)

    def draw(self):
        cv_init_it, _ = get_central_vertices('lt')
        cv_init_gs, _ = get_central_vertices('gs')
        plt.plot(cv_init_gs[:, 0], cv_init_gs[:, 1], 'b--')
        for t in range(TRACK_LEN):
            plt.plot(cv_init_it[:, 0], cv_init_it[:, 1], 'r-')
            plt.plot(cv_init_gs[:, 0], cv_init_gs[:, 1], 'b-')
            plt.plot(self.trj_solution[t, 0], self.trj_solution[t, 1], 'r*')
            plt.plot(self.estimated_inter_agent.trj_solution[t, 0],
                     self.estimated_inter_agent.trj_solution[t, 1], 'b*')
            plt.axis('equal')
            plt.xlim(5, 25)
            plt.ylim(-10, 10)
            plt.pause(0.3)
        plt.show()


# def utility_KKT(self_info, inter_info):
#     def fun(u):
#         """
#         Calculate the utility from the perspective of "self" agent
#
#         :param u: num_step by 4 array, where the 1\2 columns are control of "self" agent
#                   and 3\4 columns "inter" agent
#         :return: utility of the "self" agent under the control u
#         """
#         track_self = kinematic_model(u[0:TRACK_LEN * 2], self_info[0:3])
#         track_inter = kinematic_model(u[TRACK_LEN * 2:], inter_info[0:3])
#         track_all = [track_self[:, 0:2], track_inter[:, 0:2]]
#
#         group_cost = cal_group_cost(track_all)
#         # print(group_cost)
#         ut_self = np.cos(self_info[3]) * cal_interior_cost(track_self, self_info[4]) + \
#                   np.sin(self_info[3]) * group_cost
#         ut_inter = np.cos(inter_info[3]) * cal_interior_cost(track_inter, inter_info[4]) + \
#                    np.sin(inter_info[3]) * group_cost
#         return ut_self + ut_inter
#     return fun


def utility_IBR(self_info, inter_track):
    def fun(u):
        """
        Calculate the utility from the perspective of "self" agent

        :param u: num_step by 4 array, where the 1\2 columns are control of "self" agent
                  and 3\4 columns "inter" agent
        :return: utility of the "self" agent under the control u
        """
        p, v, h = self_info[0:3]
        init_state_4_kine = [p[0], p[1], v[0], v[1], h]
        track_info_self = kinematic_model(u, init_state_4_kine, TRACK_LEN, dt)
        track_self = track_info_self[:, 0:2]
        track_inter = inter_track
        track_all = [track_self, track_inter[:, 0:2]]
        # print(np.sin(self_info[3]))
        ut = np.cos(self_info[3]) * cal_interior_cost(track_self, self_info[4]) + \
             np.sin(self_info[3]) * cal_group_cost(track_all)
        return ut

    return fun


def cal_interior_cost(track, target):
    cv, s = get_central_vertices(target)

    # find the on-reference point of the track starting
    test = cv - track[0, 0:2]
    init_dis2cv = np.linalg.norm(test, axis=1)
    init_min_dis2cv = np.amin(init_dis2cv)
    init_index = np.where(init_min_dis2cv == init_dis2cv)

    # find the on-reference point of the track end
    end_dis2cv = np.linalg.norm(cv - track[-1, 0:2], axis=1)
    end_init_dis2cv = np.amin(end_dis2cv)
    end_index = np.where(end_init_dis2cv == end_dis2cv)

    # initialize an array to store distance from each point in the track to cv
    dis2cv = np.zeros([np.size(track, 0), 1])
    for i in range(np.size(track, 0)):
        dis2cv[i] = np.amin(np.linalg.norm(cv - track[i, 0:2], axis=1))

    "1. cost of travel delay"
    # calculate the on-reference distance of the given track (the longer the better)
    travel_distance = np.linalg.norm(track[-1, 0:2] - track[0, 0:2]) / TRACK_LEN
    cost_travel_distance = - travel_distance
    # print('cost of travel delay:', cost_travel_distance)

    "2. cost of lane deviation"
    cost_mean_deviation = dis2cv.mean()
    # print('cost of lane deviation:', cost_mean_deviation)

    "3. cost of change plan (currently not considered)"
    cost_plan_change = 0

    "overall cost"
    cost_interior = WEIGHT_DELAY * cost_travel_distance + \
                    WEIGHT_DEVIATION * cost_mean_deviation + \
                    WEIGHT_CHANGE * cost_plan_change
    # print('interior cost:', cost_interior)
    return cost_interior * WEIGHT_INT


def cal_group_cost(track_packed):
    track_self, track_inter = track_packed
    rel_distance = np.linalg.norm(track_self - track_inter, axis=1)
    min_rel_distance = np.amin(rel_distance)  # minimal distance
    min_index = np.where(min_rel_distance == rel_distance)[0]  # the time step when reach the minimal distance
    cost_group = -min_rel_distance * min_index[0] / (np.size(track_self, 0))

    # print('group cost:', cost_group)
    return cost_group * WEIGHT_GRP


# def con_inter_opt(self_info, inter_info):
#     """
#     this is the constraint for KKT method
#     :param self_info:
#     :param inter_info:
#     :return: constraint function
#     """
#
#     def con(u):
#         track_self = kinematic_model(u[0:TRACK_LEN * 2], self_info[0:3], TRACK_LEN, dt)
#         # track_inter = kinematic_model(u[TRACK_LEN * 2:], inter_info[0:3])
#         track_inter1 = kinematic_model(1.01 * u[TRACK_LEN * 2:], inter_info[0:3], TRACK_LEN, dt)
#         track_inter2 = kinematic_model(0.99 * u[TRACK_LEN * 2:], inter_info[0:3], TRACK_LEN, dt)
#
#         ut_inter1 = np.cos(inter_info[3]) * cal_interior_cost(track_inter1, inter_info[4]) + \
#                     np.sin(inter_info[3]) * cal_group_cost([track_self, track_inter1])
#         ut_inter2 = np.cos(inter_info[3]) * cal_interior_cost(track_inter2, inter_info[4]) + \
#                     np.sin(inter_info[3]) * cal_group_cost([track_self, track_inter2])
#
#         delta_ut = np.abs(ut_inter2 - ut_inter1)
#         # print('delta_ut', MAX_DELTA_UT - delta_ut)
#
#         return MAX_DELTA_UT - delta_ut
#         # return delta_ut
#
#     return con


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

    # generate LT and virtual GS agents
    agent_lt = Agent(init_position_lt, init_velocity_lt, init_heading_lt, 'lt')
    agent_gs = Agent(init_position_gs, init_velocity_gs, init_heading_gs, 'gs')
    agent_lt.estimated_inter_agent = copy.deepcopy(agent_gs)
    agent_gs.estimated_inter_agent = copy.deepcopy(agent_lt)
    agent_lt.ipv = math.pi / 3
    agent_gs.ipv = 0
    # agent_lt.trj_solution_for_inter_agent = np.repeat([init_position_gs], TRACK_LEN + 1, axis=0)
    # agent_gs.trj_solution_for_inter_agent = np.repeat([init_position_lt], TRACK_LEN + 1, axis=0)

    # KKT planning
    # track_s, track_i = agent_lt.solve_game_KKT(agent_gs)

    # IBR planning for left-turn agent
    # track_init = np.repeat([init_position_lt], TRACK_LEN + 1, axis=0)
    track_last = np.zeros_like(agent_lt.trj_solution)
    count = 0
    while np.linalg.norm(agent_lt.trj_solution[:, 0:2] - track_last[:, 0:2]) > 1e-3:
        count += 1
        print(count)
        track_last = agent_lt.trj_solution
        agent_lt.solve_game_IBR(agent_lt.estimated_inter_agent.trj_solution)
        agent_lt.estimated_inter_agent.solve_game_IBR(agent_lt.trj_solution)
        if count > 30:
            break

    cv_init_it, _ = get_central_vertices('lt')
    cv_init_gs, _ = get_central_vertices('gs')
    plt.plot(cv_init_gs[:, 0], cv_init_gs[:, 1], 'b--')
    for t in range(TRACK_LEN):
        plt.plot(cv_init_it[:, 0], cv_init_it[:, 1], 'r-')
        plt.plot(cv_init_gs[:, 0], cv_init_gs[:, 1], 'b-')
        plt.plot(agent_lt.trj_solution[t, 0], agent_lt.trj_solution[t, 1], 'r*')
        plt.plot(agent_lt.estimated_inter_agent.trj_solution[t, 0],
                 agent_lt.estimated_inter_agent.trj_solution[t, 1], 'b*')
        plt.axis('equal')
        plt.xlim(5, 25)
        plt.ylim(-10, 10)
        plt.pause(0.3)
    plt.show()
