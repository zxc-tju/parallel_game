import copy
import math
import numpy as np
from matplotlib import pyplot as plt
from agent import Agent
from tools.utility import get_central_vertices
import time
import pickle


class Scenario:
    def __init__(self, pos, vel, heading, ipv):
        self.position = {'lt': np.array(pos[0]), 'gs': np.array(pos[1])}
        self.velocity = {'lt': np.array(vel[0]), 'gs': np.array(vel[1])}
        self.heading = {'lt': np.array(heading[0]), 'gs': np.array(heading[1])}
        self.ipv = {'lt': np.array(ipv[0]), 'gs': np.array(ipv[1])}


class Simulator:
    def __init__(self, version):
        self.semantic_result = None
        self.version = version
        self.output_directory = './outputs/simulation/version' + str(self.version)
        self.tag = None
        self.scenario = None
        self.agent_lt = None
        self.agent_gs = None
        self.num_step = 0
        self.ending_point = None

    def initialize(self, scenario, case_tag):
        self.scenario = scenario
        self.agent_lt = Agent(scenario.position['lt'], scenario.velocity['lt'], scenario.heading['lt'], 'lt')
        self.agent_gs = Agent(scenario.position['gs'], scenario.velocity['gs'], scenario.heading['gs'], 'gs')
        self.agent_lt.estimated_inter_agent = copy.deepcopy(self.agent_gs)
        self.agent_gs.estimated_inter_agent = copy.deepcopy(self.agent_lt)
        self.agent_lt.ipv = self.scenario.ipv['lt']
        self.agent_gs.ipv = self.scenario.ipv['gs']
        self.tag = case_tag

    def ibr_iteration(self, num_step, iter_limit):
        self.num_step = num_step
        for t in range(self.num_step):
            print('time_step: ', t, '/', self.num_step)

            "==plan for left-turn=="
            # ==interaction with parallel virtual agents
            self.agent_lt.interact_with_parallel_virtual_agents(self.agent_gs, iter_limit)

            # ==interaction with estimated agent
            self.agent_lt.interact_with_estimated_agents(iter_limit)

            "==plan for go straight=="
            # ==interaction with parallel virtual agents
            self.agent_gs.interact_with_parallel_virtual_agents(self.agent_lt, iter_limit)

            # ==interaction with estimated agent
            self.agent_gs.interact_with_estimated_agents(iter_limit)

            "==update state=="
            self.agent_lt.update_state(self.agent_gs, 1)
            self.agent_gs.update_state(self.agent_lt, 1)

    def save_data(self):
        filename = self.output_directory + '/data/agents_info' \
                   + 'case_' + str(self.tag) \
                   + '.pckl'
        f = open(filename, 'wb')
        pickle.dump([self.agent_lt, self.agent_gs, self.semantic_result, self.tag, self.ending_point], f)
        f.close()
        print('case_' + str(self.tag), ' saved')

    def post_process(self):
        track_lt = self.agent_lt.observed_trajectory
        track_gs = self.agent_gs.observed_trajectory
        pos_delta = track_gs - track_lt

        "whether the LT vehicle yield"
        pos_x_smaller = pos_delta[pos_delta[:, 0] < 0]
        pos_y_larger = pos_x_smaller[pos_x_smaller[:, 1] > 0]
        yield_points = np.size(pos_y_larger, 0)
        if yield_points:
            self.semantic_result = 'yield'

            "where the interaction finish"
            ind_coll = np.where(pos_y_larger[0, 0] == pos_delta[:, 0])
            ind = ind_coll[0] - 1
            self.ending_point = {'lt': self.agent_lt.observed_trajectory[ind, :],
                                 'gs': self.agent_gs.observed_trajectory[ind, :]}

            print('LT vehicle yielded. \n')
            print('interaction finished at No.' + str(ind+1) + ' frame\n')
            print('GS info:' + str(self.ending_point['gs']) + '\n')
            print('LT info:' + str(self.ending_point['lt']) + '\n')
            print('px py vx vy heading')

        else:
            self.semantic_result = 'rush'
            print('LT vehicle rushed. \n')

    def visualize(self):
        cv_it, _ = get_central_vertices('lt')
        cv_gs, _ = get_central_vertices('gs')

        # set figures
        fig = plt.figure(figsize=(12, 4))
        fig.suptitle('case_' + str(self.tag))

        "====show plans at each time step===="
        ax1 = fig.add_subplot(131, title='trajectory_LT_' + self.semantic_result)
        # central vertices
        ax1.plot(cv_it[:, 0], cv_it[:, 1], 'r-')
        ax1.plot(cv_gs[:, 0], cv_gs[:, 1], 'b-')

        # position at each time step
        ax1.scatter(self.agent_lt.observed_trajectory[:, 0],
                    self.agent_lt.observed_trajectory[:, 1],
                    s=100,
                    alpha=0.6,
                    color='red',
                    label='left-turn')
        ax1.scatter(self.agent_gs.observed_trajectory[:, 0],
                    self.agent_gs.observed_trajectory[:, 1],
                    s=100,
                    alpha=0.6,
                    color='blue',
                    label='go-straight')

        # full tracks at each time step
        for t in range(self.num_step):
            lt_track = self.agent_lt.trj_solution_collection[t]
            ax1.plot(lt_track[:, 0], lt_track[:, 1], '--', color='red')
            gs_track = self.agent_gs.trj_solution_collection[t]
            ax1.plot(gs_track[:, 0], gs_track[:, 1], '--', color='blue')

        # connect two agents
        for t in range(self.num_step + 1):
            ax1.plot([self.agent_lt.observed_trajectory[t, 0], self.agent_gs.observed_trajectory[t, 0]],
                     [self.agent_lt.observed_trajectory[t, 1], self.agent_gs.observed_trajectory[t, 1]],
                     color='black',
                     alpha=0.2)

        ax1.set(xlim=[5, 25], ylim=[-15, 15])

        "====show IPV and uncertainty===="
        ax2 = fig.add_subplot(132, title='ipv')
        x_range = np.array(range(len(self.agent_lt.estimated_inter_agent.ipv_collection)))
        y_lt = np.array(self.agent_gs.estimated_inter_agent.ipv_collection)
        y_gs = np.array(self.agent_lt.estimated_inter_agent.ipv_collection)

        # actual ipv
        ax2.plot(x_range, self.agent_lt.ipv * np.ones_like(x_range),
                 color='red',
                 linewidth=5,
                 label='actual lt IPV')
        ax2.plot(x_range, self.agent_gs.ipv * np.ones_like(x_range),
                 color='blue',
                 linewidth=5,
                 label='actual gs IPV')

        # estimated ipv
        ax2.plot(x_range, y_lt, color='red', label='estimated lt IPV')
        ax2.plot(x_range, y_gs, color='blue', label='estimated gs IPV')

        # error bar
        y_error_lt = np.array(self.agent_lt.estimated_inter_agent.ipv_error_collection)
        ax2.fill_between(x_range, y_lt - y_error_lt, y_lt + y_error_lt,
                         alpha=0.3,
                         color='red',
                         label='gs IPV error')
        y_error_gs = np.array(self.agent_gs.estimated_inter_agent.ipv_error_collection)
        ax2.fill_between(x_range, y_gs - y_error_gs, y_gs + y_error_gs,
                         alpha=0.3,
                         color='blue',
                         label='lt IPV error')

        "====show velocity===="
        ax3 = fig.add_subplot(133, title='velocity')
        x_range = np.array(range(np.size(self.agent_lt.observed_trajectory, 0)))
        vel_norm_lt = np.linalg.norm(self.agent_lt.observed_trajectory[:, 2:4], axis=1)
        vel_norm_gs = np.linalg.norm(self.agent_gs.observed_trajectory[:, 2:4], axis=1)
        ax3.plot(x_range, vel_norm_lt, color='red', label='LT velocity')
        ax3.plot(x_range, vel_norm_gs, color='blue', label='FC velocity')

        ax1.legend()
        ax2.legend()
        ax3.legend()

        plt.savefig(self.output_directory + '/figures/'
                    + 'case_' + str(self.tag) + '.png')
        # plt.show()


def main1():
    """
    ==== main for simulating unprotected left-turning ====

    1. model type is controlled by ipv and iteration number (3: VGIM , 0: Optimal controller)

    2. manual Continuous Interaction: if LT yielded, print the ending point of the interaction and
    the next interaction starts at the ending point  (used for setting LT vehicle's initial state)

    3. change FC vehicle' ipv and initial state, and scenario tag for each simulation

    4. simulation results are saved in simulation/version28
    :return:
    """
    # tag = 'test'

    # r = 5
    # c = 6
    # tag = 'round2-' + str(r)+'-' + str(c)

    tag = 'round3-OPT-safe-gs-1'  # TODO

    # initial state of the left-turn vehicle
    init_position_lt = [11, -5.8]  # TODO
    init_velocity_lt = [0.25, 0.28]  # TODO
    init_heading_lt = math.pi / 4  # TODO
    ipv_lt = math.pi / 8
    # initial state of the go-straight vehicle
    init_position_gs = [22, -2]  # TODO
    init_velocity_gs = [-5, 0]
    init_heading_gs = math.pi
    ipv_gs = math.pi / 4  # TODO

    simu_scenario = Scenario([init_position_lt, init_position_gs],
                             [init_velocity_lt, init_velocity_gs],
                             [init_heading_lt, init_heading_gs],
                             [ipv_lt, ipv_gs])

    simu = Simulator(28)
    simu.initialize(simu_scenario, tag)

    steps = 20
    iterations = 3
    simu.ibr_iteration(steps, iterations)
    simu.post_process()
    simu.save_data()
    simu.visualize()


def main2():
    """
    ==== main for simulating RANDOM* unprotected left-turning ====

    1. model type is controlled by ipv and iteration number (3: VGIM , 1: Optimal controller)

    2. Continuous Interaction: if LT yielded, print the ending point of the interaction and
    the next interaction starts at the ending point

    3. change tag for each simulation

    4. simulation results are saved in simulation/version29

    5*. straight-through traffic are endless and generated with random ipv and gap
    :return:
    """
    print('start main for random interaction')


if __name__ == '__main__':
    "无保护左转实验- 多模型对比"
    main1()

    "无保护左转实验- 随机交互"
    main2()

