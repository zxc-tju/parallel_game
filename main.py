import copy
import math
import numpy as np
from matplotlib import pyplot as plt
from agent import Agent
from tools.utility import get_central_vertices
import time
import pickle

final_illustration_needed = 0
in_loop_illustration_needed = 0
num_step = 20

# INITIAL_IPV_LEFT_TURN = math.pi / 4
# INITIAL_IPV_GO_STRAIGHT = 0 * math.pi / 4
virtual_agent_IPV_range = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]) * math.pi / 9


def multi_simulate(process_id, gs_ipv_set, lt_ipv_set):
    for gs_ipv in gs_ipv_set:
        for lt_ipv in lt_ipv_set:
            simulate(gs_ipv, lt_ipv)
            # print('go straight: ', gs_ipv)
            # print('left turn: ', lt_ipv)
    print('#======process: ', process_id, 'finished')


def simulate(gs_ipv, lt_ipv):
    # initial state of the left-turn vehicle
    init_position_lt = np.array([13, -7])
    init_velocity_lt = np.array([2, 0.3])
    init_heading_lt = math.pi / 4
    # initial state of the go-straight vehicle
    init_position_gs = np.array([18, -2])
    init_velocity_gs = np.array([0, 0])
    init_heading_gs = math.pi

    # generate LT and GS agents
    agent_lt = Agent(init_position_lt, init_velocity_lt, init_heading_lt, 'lt')
    agent_gs = Agent(init_position_gs, init_velocity_gs, init_heading_gs, 'gs')
    # generate estimated interacting agent of LT and GS agents respectively
    agent_lt.estimated_inter_agent = copy.deepcopy(agent_gs)
    agent_gs.estimated_inter_agent = copy.deepcopy(agent_lt)
    # initialize IPV
    # agent_lt.ipv = INITIAL_IPV_LEFT_TURN
    # agent_gs.ipv = INITIAL_IPV_GO_STRAIGHT
    agent_lt.ipv = lt_ipv * math.pi / 9
    agent_gs.ipv = gs_ipv * math.pi / 9

    "====IRB process===="
    for t in range(num_step):
        # print('time_step: ', t, '/', num_step)

        "==plan for left-turn=="
        # ==interaction with parallel virtual agents
        virtual_gs_track_collection = []
        for ipv_temp in virtual_agent_IPV_range:

            count_lt = 0  # count number of iteration
            virtual_inter_gs = copy.deepcopy(agent_gs)
            virtual_inter_gs.ipv = ipv_temp
            agent_lt_temp = copy.deepcopy(agent_lt)
            track_last_lt_temp = np.zeros_like(agent_lt.trj_solution)  # initialize a track reservation

            while np.linalg.norm(agent_lt_temp.trj_solution[:, 0:2] - track_last_lt_temp[:, 0:2]) > 1e-3:
                count_lt += 1
                track_last_lt_temp = agent_lt_temp.trj_solution
                agent_lt_temp.solve_game_IBR(virtual_inter_gs.trj_solution)
                virtual_inter_gs.solve_game_IBR(agent_lt_temp.trj_solution)
                if count_lt > 10:  # limited to less than 10 iterations
                    break
            virtual_gs_track_collection.append(virtual_inter_gs.trj_solution)
        agent_lt.estimated_inter_agent.virtual_track_collection.append(virtual_gs_track_collection)

        # ==interaction with estimated agent
        count_lt = 0  # count number of iteration
        track_last_lt = np.zeros_like(agent_lt.trj_solution)  # initialize a track reservation
        while np.linalg.norm(agent_lt.trj_solution[:, 0:2] - track_last_lt[:, 0:2]) > 1e-3:
            count_lt += 1
            track_last_lt = agent_lt.trj_solution
            agent_lt.solve_game_IBR(agent_lt.estimated_inter_agent.trj_solution)
            agent_lt.estimated_inter_agent.solve_game_IBR(agent_lt.trj_solution)
            if count_lt > 10:  # limited to less than 10 iterations
                break

        if in_loop_illustration_needed:
            agent_lt.draw()

        "==plan for go straight=="
        count_gs = 0  # count number of iteration
        track_last_gs = np.zeros_like(agent_gs.trj_solution)  # initialize a track reservation

        # ==interaction with parallel virtual agents
        virtual_lt_track_collection = []
        for ipv_temp in virtual_agent_IPV_range:
            # print('idx: ', ipv_temp)
            virtual_inter_lt = copy.deepcopy(agent_lt)
            agent_gs_temp = copy.deepcopy(agent_gs)
            virtual_inter_lt.ipv = ipv_temp

            while np.linalg.norm(agent_gs_temp.trj_solution[:, 0:2] - track_last_gs[:, 0:2]) > 1e-3:
                count_gs += 1
                track_last_gs = agent_gs_temp.trj_solution
                agent_gs_temp.solve_game_IBR(virtual_inter_lt.trj_solution)
                virtual_inter_lt.solve_game_IBR(agent_gs_temp.trj_solution)
                if count_gs > 10:
                    count_gs = 0
                    break
            virtual_lt_track_collection.append(virtual_inter_lt.trj_solution)
        agent_gs.estimated_inter_agent.virtual_track_collection.append(virtual_lt_track_collection)

        # ==interaction with estimated agent
        count_gs = 0  # count number of iteration
        track_last_gs = np.zeros_like(agent_gs.trj_solution)
        while np.linalg.norm(agent_gs.trj_solution[:, 0:2] - track_last_gs[:, 0:2]) > 1e-3:
            count_gs += 1
            track_last_gs = agent_gs.trj_solution
            agent_gs.solve_game_IBR(agent_gs.estimated_inter_agent.trj_solution)
            agent_gs.estimated_inter_agent.solve_game_IBR(agent_gs.trj_solution)
            if count_gs > 10:
                break

        if in_loop_illustration_needed:
            agent_gs.draw()

        # update state
        agent_lt.update_state(agent_gs, virtual_agent_IPV_range)
        agent_gs.update_state(agent_lt, virtual_agent_IPV_range)
        print("estimated gs ipv:", agent_lt.estimated_inter_agent.ipv)
        print("estimated lt ipv:", agent_gs.estimated_inter_agent.ipv)

    "====save data===="
    filename = './outputs/version6/agents_info' + '_gs_' + str(gs_ipv) + '_lt_' + str(lt_ipv) + '_math.pi_9' + '.pckl'
    f = open(filename, 'wb')
    pickle.dump([agent_lt, agent_gs], f)
    f.close()
    print('gs_' + str(gs_ipv) + '_lt_' + str(lt_ipv), 'saved')

    "====visualization===="
    if final_illustration_needed:
        plt.figure()
        cv_init_it, _ = get_central_vertices('lt')
        cv_init_gs, _ = get_central_vertices('gs')
        for t in range(num_step):
            # central vertices
            plt.plot(cv_init_it[:, 0], cv_init_it[:, 1], 'r-')
            plt.plot(cv_init_gs[:, 0], cv_init_gs[:, 1], 'b-')

            # full tracks at each time step
            lt_track = agent_lt.trj_solution_collection[t]
            plt.plot(lt_track[:, 0], lt_track[:, 1], '--')
            gs_track = agent_gs.trj_solution_collection[t]
            plt.plot(gs_track[:, 0], gs_track[:, 1], '--')

        plt.axis('equal')
        plt.xlim(5, 25)
        plt.ylim(-15, 15)
        # plt.show()

        plt.figure()
        # central vertices
        plt.plot(cv_init_it[:, 0], cv_init_it[:, 1], 'r-')
        plt.plot(cv_init_gs[:, 0], cv_init_gs[:, 1], 'b-')
        # position at each time step
        plt.scatter(agent_lt.trajectory[:, 0],
                    agent_lt.trajectory[:, 1],
                    s=100,
                    alpha=0.6,
                    color='red',
                    label='left-turn')
        plt.scatter(agent_gs.trajectory[:, 0],
                    agent_gs.trajectory[:, 1],
                    s=100,
                    alpha=0.6,
                    color='blue',
                    label='go-straight')

        for t in range(len(agent_lt.trajectory)):
            plt.plot([agent_lt.trajectory[t, 0], agent_gs.trajectory[t, 0]],
                     [agent_lt.trajectory[t, 1], agent_gs.trajectory[t, 1]], color='black')
        plt.axis('equal')
        plt.xlim(5, 25)
        plt.ylim(-15, 15)

        plt.figure()

        x_range = np.array(range(len(agent_lt.estimated_inter_agent.ipv_collection)))
        y_lt = np.array(agent_lt.estimated_inter_agent.ipv_collection)
        y_error_lt = np.array(agent_lt.estimated_inter_agent.ipv_error_collection)
        plt.fill_between(x_range, y_lt - y_error_lt, y_lt + y_error_lt,
                         alpha=0.4,
                         color='blue',
                         label='estimated gs IPV')
        plt.plot(x_range, gs_ipv * math.pi / 9 * np.ones_like(x_range), label='actual gs IPV')

        y_gs = np.array(agent_gs.estimated_inter_agent.ipv_collection)
        y_error_gs = np.array(agent_gs.estimated_inter_agent.ipv_error_collection)
        plt.fill_between(x_range, y_gs - y_error_gs, y_gs + y_error_gs,
                         alpha=0.4,
                         color='red',
                         label='estimated lt IPV')
        plt.plot(x_range, lt_ipv * math.pi / 9 * np.ones_like(x_range), label='actual lt IPV')


if __name__ == '__main__':
    from multiprocessing import Process

    tic = time.perf_counter()
    lt_ipv_set_full = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    processes = [Process(target=multi_simulate, args=(1, [-4], lt_ipv_set_full)),
                 Process(target=multi_simulate, args=(2, [-3], lt_ipv_set_full)),
                 Process(target=multi_simulate, args=(3, [-2], lt_ipv_set_full)),
                 Process(target=multi_simulate, args=(4, [-1], lt_ipv_set_full)),
                 Process(target=multi_simulate, args=(5, [0], lt_ipv_set_full)),
                 Process(target=multi_simulate, args=(6, [1], lt_ipv_set_full)),
                 Process(target=multi_simulate, args=(7, [2], lt_ipv_set_full)),
                 Process(target=multi_simulate, args=(8, [3], lt_ipv_set_full)),
                 Process(target=multi_simulate, args=(9, [4], lt_ipv_set_full)),
                 ]
    [p.start() for p in processes]  # 开启进程
    [p.join() for p in processes]  # 等待进程依次结束

    # for gs_ipv in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
    #     for lt_ipv in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:

    # for gs_ipv in [1]:
    #     for lt_ipv in [1]:
    #         simulate(gs_ipv, lt_ipv)

    toc = time.perf_counter()

    print(f"whole process takes {toc - tic:0.4f} seconds")
