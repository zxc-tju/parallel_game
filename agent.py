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
WEIGHT_STEERING = 0.1
weight_metric = np.array([WEIGHT_DELAY, WEIGHT_DEVIATION, WEIGHT_STEERING])
weight_metric = weight_metric / weight_metric.sum()

# parameters of action bounds
MAX_STEERING_ANGLE = math.pi / 6
MAX_ACCELERATION = 3.0

# initial guess on interacting agent's IPV
INITIAL_IPV_GUESS = 0
virtual_agent_IPV_range = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]) * math.pi / 9

# weight of interior and group cost
WEIGHT_INT = 1
WEIGHT_GRP = 0.9

# likelihood function
sigma = 0.1


class Agent:
    def __init__(self, position, velocity, heading, target):
        self.position = position
        self.velocity = velocity
        self.heading = heading
        self.target = target
        # conducted trajectory
        self.observed_trajectory = np.array([[self.position[0],
                                              self.position[1],
                                              self.velocity[0],
                                              self.velocity[1],
                                              self.heading], ])
        self.trj_solution = np.repeat([position], TRACK_LEN, axis=0)
        self.trj_solution_collection = []
        self.action = []
        self.action_collection = []
        self.estimated_inter_agent = None
        self.ipv = 0
        self.ipv_error = None
        self.ipv_collection = []
        self.ipv_error_collection = []
        self.virtual_track_collection = []

    def solve_game_IBR(self, inter_track):
        track_len = np.size(inter_track, 0)
        self_info = [self.position,
                     self.velocity,
                     self.heading,
                     self.ipv,
                     self.target]

        p, v, h = self_info[0:3]
        init_state_4_kine = [p[0], p[1], v[0], v[1], h]  # initial state
        fun = utility_IBR(self_info, inter_track)  # objective function
        u0 = np.concatenate([1*np.ones([(track_len - 1), 1]),
                             np.zeros([(track_len - 1), 1])])  # initialize solution
        bds = [(-MAX_ACCELERATION, MAX_ACCELERATION) for i in range(track_len - 1)] + \
              [(-MAX_STEERING_ANGLE, MAX_STEERING_ANGLE) for i in range(track_len - 1)]  # boundaries

        res = minimize(fun, u0, bounds=bds, method='SLSQP')
        x = np.reshape(res.x, [2, track_len - 1]).T
        self.action = x
        self.trj_solution = kinematic_model(x, init_state_4_kine, track_len, dt)
        return self.trj_solution

    def interact_with_parallel_virtual_agents(self, agent_inter):
        """
        generate copy of the interacting agent and interact with them
        :param agent_inter: Agent:interacting agent
        :return:
        """
        virtual_agent_track_collection = []

        for ipv_temp in virtual_agent_IPV_range:
            virtual_inter_agent = copy.deepcopy(agent_inter)
            virtual_inter_agent.ipv = ipv_temp
            agent_self_temp = copy.deepcopy(self)

            count_iter = 0  # count number of iteration
            last_self_track = np.zeros_like(self.trj_solution)  # initialize a track reservation
            while np.linalg.norm(agent_self_temp.trj_solution[:, 0:2] - last_self_track[:, 0:2]) > 1e-3:
                count_iter += 1
                last_self_track = agent_self_temp.trj_solution
                agent_self_temp.solve_game_IBR(virtual_inter_agent.trj_solution)
                virtual_inter_agent.solve_game_IBR(agent_self_temp.trj_solution)
                if count_iter > 10:
                    count_iter = 0
                    break
            virtual_agent_track_collection.append(virtual_inter_agent.trj_solution)
        self.estimated_inter_agent.virtual_track_collection.append(virtual_agent_track_collection)

    def interact_with_estimated_agents(self):
        """
        interact with the estimated interacting agent. this agent's IPV is continuously updated.
        :return:
        """
        count_iter = 0  # count number of iteration
        last_self_track = np.zeros_like(self.trj_solution)  # initialize a track reservation
        while np.linalg.norm(self.trj_solution[:, 0:2] - last_self_track[:, 0:2]) > 1e-3:
            count_iter += 1
            last_self_track = self.trj_solution
            self.solve_game_IBR(self.estimated_inter_agent.trj_solution)
            self.estimated_inter_agent.solve_game_IBR(self.trj_solution)
            if count_iter > 10:  # limited to less than 10 iterations
                break

    def update_state(self, inter_agent, method):
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
        self.observed_trajectory = np.concatenate((self.observed_trajectory, new_track_point), axis=0)

        self.trj_solution_collection.append(self.trj_solution)
        self.action_collection.append(self.action)

        # update IPV
        current_time = np.size(self.observed_trajectory, 0) - 2
        if current_time > 1:
            start_time = max(0, current_time - 6)
            time_duration = current_time - start_time

            if method == 1:
                "====parallel game method===="
                candidates = self.estimated_inter_agent.virtual_track_collection[start_time]
                virtual_track_collection = []
                for i in range(len(candidates)):
                    virtual_track_collection.append(candidates[i][0:time_duration, 0:2])
                actual_track = inter_agent.observed_trajectory[start_time:current_time, 0:2]

                ipv_weight = cal_reliability(actual_track, virtual_track_collection)

                # weighted sum of all candidates' IPVs
                self.estimated_inter_agent.ipv = sum(virtual_agent_IPV_range * ipv_weight)

                # save updated ipv and estimation error
                self.estimated_inter_agent.ipv_collection.append(self.estimated_inter_agent.ipv)
                error = 1 - np.sqrt(sum(ipv_weight ** 2))
                self.estimated_inter_agent.ipv_error_collection.append(error)
                "====end of parallel game method===="

            elif method == 2:
                "====rational perspective method===="

                # rearrange conducted actions at each time step
                u_list_self = []
                u_list_inter = []
                for i in range(start_time, current_time):
                    u_list_self.append([self.action_collection[i][0, 0], self.action_collection[i][0, 1]])
                    u_list_inter.append(
                        [inter_agent.action_collection[i][0, 0], inter_agent.action_collection[i][0, 1]])

                # initial state
                init_state_4_kine_self = self.observed_trajectory[start_time, :]
                init_state_4_kine_inter = inter_agent.observed_trajectory[start_time, :]

                u_conducted_self = np.array(u_list_self)
                u_conducted_inter = np.array(u_list_inter)

                # info at margin1
                u_margin_self1 = u_conducted_self * 1.01
                u_margin_inter1 = u_conducted_inter * 1.01
                track_margin_inter1 = kinematic_model(u_margin_inter1, init_state_4_kine_inter, time_duration + 1, dt)
                track_margin_self1 = kinematic_model(u_margin_self1, init_state_4_kine_self, time_duration + 1, dt)
                cost_inerior_margin1 = cal_interior_cost(u_margin_inter1,
                                                        track_margin_inter1[:, 0:2],
                                                        inter_agent.target)
                cost_group_margin1 = cal_group_cost([track_margin_inter1[:, 0:2], track_margin_self1[:, 0:2]])

                # # info at margin2
                # u_margin_self2 = u_conducted_self * 0.999
                # u_margin_inter2 = u_conducted_inter * 0.999
                # track_margin_inter2 = kinematic_model(u_margin_inter2, init_state_4_kine_inter, time_duration + 1, dt)
                # track_margin_self2 = kinematic_model(u_margin_self2, init_state_4_kine_self, time_duration + 1, dt)
                # cost_inerior_margin2 = cal_interior_cost(u_margin_inter2,
                #                                          track_margin_inter2[:, 0:2],
                #                                          inter_agent.target)
                # cost_group_margin2 = cal_group_cost([track_margin_inter2[:, 0:2], track_margin_self2[:, 0:2]])
                # # atan
                # self.estimated_inter_agent.ipv = - math.atan((cost_inerior_margin1 - cost_inerior_margin2)
                #                                              / (cost_group_margin1 - cost_group_margin2))

                # info at actual solution
                track_actual_inter = inter_agent.observed_trajectory[start_time:current_time + 1, 0:2]
                track_actual_self = self.observed_trajectory[start_time:current_time + 1, 0:2]
                cost_inerior = cal_interior_cost(u_conducted_inter,
                                                 track_actual_inter,
                                                 inter_agent.target)
                cost_group = cal_group_cost([track_actual_inter, track_actual_self])
                # atan
                self.estimated_inter_agent.ipv = - math.atan((cost_inerior - cost_inerior_margin1)
                                                             / (cost_group - cost_group_margin1))
                # save updated ipv and estimation error
                self.estimated_inter_agent.ipv_collection.append(self.estimated_inter_agent.ipv)
                "====end of rational perspective method===="

    def estimate_self_ipv_in_NDS(self, self_actual_track, inter_track):
        self_virtual_track_collection = []
        # ipv_range = np.random.normal(self.ipv, math.pi/6, 6)
        ipv_range = virtual_agent_IPV_range
        for ipv_temp in ipv_range:
            agent_self_temp = copy.deepcopy(self)
            agent_self_temp.ipv = ipv_temp
            # generate track with varied ipv
            virtual_track_temp = agent_self_temp.solve_game_IBR(inter_track)
            # save track into a collection
            self.virtual_track_collection.append(virtual_track_temp[:, 0:2])

        # calculate reliability of each track
        ipv_weight = cal_reliability(self_actual_track, self.virtual_track_collection)

        # weighted sum of all candidates' IPVs
        self.ipv = sum(ipv_range * ipv_weight)
        self.ipv_error = 1 - np.sqrt(sum(ipv_weight ** 2))

        # # save updated ipv and estimation error
        # self.ipv_collection.append(self.ipv)
        # self.ipv_error_collection.append(self.ipv_error)

    def draw(self):
        cv_init_it, _ = get_central_vertices('lt')
        cv_init_gs, _ = get_central_vertices('gs')
        plt.plot(cv_init_gs[:, 0], cv_init_gs[:, 1], 'b--')
        for t in range(np.size(self.trj_solution, 0)):
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


def utility_IBR(self_info, track_inter):
    def fun(u):
        """
        Calculate the utility from the perspective of "self" agent

        :param u: num_step by 4 array, where the 1\2 columns are control of "self" agent
                  and 3\4 columns "inter" agent
        :return: utility of the "self" agent under the control u
        """
        p, v, h = self_info[0:3]
        init_state_4_kine = [p[0], p[1], v[0], v[1], h]
        track_info_self = kinematic_model(u, init_state_4_kine, np.size(track_inter, 0), dt)
        track_self = track_info_self[:, 0:2]
        track_all = [track_self, track_inter[:, 0:2]]
        # print(np.sin(self_info[3]))
        util = np.cos(self_info[3]) * cal_interior_cost(u, track_self, self_info[4]) + \
               np.sin(self_info[3]) * cal_group_cost(track_all)
        return util

    return fun


def cal_interior_cost(action, track, target):
    cv, s = get_central_vertices(target)

    # find the on-reference point of the track starting
    test = cv - track[0, 0:2]
    init_dis2cv = np.linalg.norm(test, axis=1)
    init_min_dis2cv = np.amin(init_dis2cv)
    init_index = np.where(init_min_dis2cv == init_dis2cv)

    # find the on-reference point of the track end
    # end_dis2cv = np.linalg.norm(cv - track[-1, 0:2], axis=1)
    # end_init_dis2cv = np.amin(end_dis2cv)
    # end_index = np.where(end_init_dis2cv == end_dis2cv)

    # initialize an array to store distance from each point in the track to cv
    dis2cv = np.zeros([np.size(track, 0), 1])
    for i in range(np.size(track, 0)):
        dis2cv[i] = np.amin(np.linalg.norm(cv - track[i, 0:2], axis=1))

    "1. cost of travel delay"
    # calculate the on-reference distance of the given track (the longer the better)
    travel_distance = np.linalg.norm(track[-1, 0:2] - track[0, 0:2]) / np.size(track, 0)
    cost_travel_distance = - travel_distance
    # print('cost of travel delay:', cost_travel_distance)

    "2. cost of lane deviation"
    cost_mean_deviation = max(0, dis2cv.mean() - 0.3)
    # print('cost of lane deviation:', cost_mean_deviation)

    "3. cost of steering"
    cost_steering = 0
    # if target in {'gs_nds', 'gs'} and action is not []:
    #     delta_slice = slice(int(np.size(action, 0) / 2), np.size(action, 0))
    #     delta = action[delta_slice]
    #     cost_steering = np.sum(np.abs(delta)) / np.size(track, 0)

    cost_metric = np.array([cost_travel_distance, cost_mean_deviation, cost_steering])

    "overall cost"
    cost_interior = weight_metric.dot(cost_metric.T)
    # cost_interior = WEIGHT_DELAY * cost_travel_distance + \
    #                 WEIGHT_DEVIATION * cost_mean_deviation + \
    #                 WEIGHT_STEERING * cost_steering

    return cost_interior * WEIGHT_INT


def cal_group_cost(track_packed):
    track_self, track_inter = track_packed
    rel_distance = np.linalg.norm(track_self - track_inter, axis=1)
    "version 1"
    min_rel_distance = np.amin(rel_distance)  # minimal distance
    min_index = np.where(min_rel_distance == rel_distance)[0]  # the time step when reach the minimal distance
    cost_group1 = -min_rel_distance * min_index[0] / (np.size(track_self, 0)) / rel_distance[0]

    "version 2"
    # rel_speed = (rel_distance[1:] - rel_distance[0:-1]) / dt
    # rel_speed[np.where(rel_speed > 0)] = 0
    # cost_group2 = -np.sum(rel_speed) / (np.size(track_self, 0))

    # print('group cost:', cost_group)
    return cost_group1 * WEIGHT_GRP


def cal_reliability(act_trck, vir_trck_coll):
    """

    :param act_trck: actual_track
    :param vir_trck_coll: virtual_track_collection
    :return:
    """

    candidates_num = len(vir_trck_coll)
    var = np.zeros(candidates_num)
    for i in range(candidates_num):
        virtual_track = vir_trck_coll[i]
        rel_dis = np.linalg.norm(virtual_track - act_trck, axis=1)  # distance vector
        var[i] = np.power(
            np.prod(
                (1 / sigma / np.sqrt(2 * math.pi))
                * np.exp(- rel_dis ** 2 / (2 * sigma ** 2))
            )
            , 1 / np.size(act_trck, 0))

        if var[i] < 0:
            var[i] = 0

    weight = var / (sum(var))
    # print(weight)
    return weight


if __name__ == '__main__':
    "test kinematic_model"
    # action = np.concatenate((1 * np.ones((TRACK_LEN, 1)),  0.3 * np.ones((TRACK_LEN, 1))), axis=1)
    # position = np.array([0, 0])
    # velocity = np.array([3, 3])
    # heading = np.array([math.pi/4])
    # init_info = [position, velocity, heading]
    # observed_trajectory = kinematic_model(action, init_info)
    # x = observed_trajectory[:, 0]
    # y = observed_trajectory[:, 1]
    # plt.figure()
    # plt.plot(x, y, 'r-')
    # # plt.plot(observed_trajectory[:, 3], 'r-')
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
