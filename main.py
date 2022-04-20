import copy
import math
import numpy as np
from matplotlib import pyplot as plt
from agent import Agent
from tools.utility import get_central_vertices
import time
import pickle

ipv_update_method = 1
"""
notes for ipv_update_method:
1: parallel game method
2: rational perspective method
"""

in_loop_illustration_needed = 0
num_step = 30

"*****Check below before run!!!*****"
output_directory = './outputs/simulation/version26'
final_illustration_needed = 1
save_data_needed = 1


def simulate(gs_ipv_sim, lt_ipv_sim):
    # initial state of the left-turn vehicle
    init_position_lt = np.array([11, -5.8])
    init_velocity_lt = np.array([1.5, 0.3])
    init_heading_lt = math.pi / 4
    # initial state of the go-straight vehicle
    init_position_gs = np.array([18, -2])
    init_velocity_gs = np.array([-1, 0])
    init_heading_gs = math.pi

    # generate LT and GS agents
    agent_lt = Agent(init_position_lt, init_velocity_lt, init_heading_lt, 'lt')
    agent_gs = Agent(init_position_gs, init_velocity_gs, init_heading_gs, 'gs')
    # generate estimated interacting agent of LT and GS agents respectively
    agent_lt.estimated_inter_agent = copy.deepcopy(agent_gs)
    agent_gs.estimated_inter_agent = copy.deepcopy(agent_lt)
    # initialize IPV
    # agent_lt.ipv = lt_ipv_sim * math.pi / 8
    # agent_gs.ipv = gs_ipv_sim * math.pi / 8

    agent_lt.ipv = lt_ipv_sim
    agent_gs.ipv = gs_ipv_sim

    "====IRB process===="
    for t in range(num_step):
        print('time_step: ', t, '/', num_step)

        "==plan for left-turn=="
        # ==interaction with parallel virtual agents
        if ipv_update_method == 1:
            agent_lt.interact_with_parallel_virtual_agents(agent_gs)

        # ==interaction with estimated agent
        agent_lt.interact_with_estimated_agents()

        if in_loop_illustration_needed:
            agent_lt.draw()

        "==plan for go straight=="
        # ==interaction with parallel virtual agents
        if ipv_update_method == 1:
            agent_gs.interact_with_parallel_virtual_agents(agent_lt)

        # ==interaction with estimated agent
        agent_gs.interact_with_estimated_agents()

        if in_loop_illustration_needed:
            agent_gs.draw()

        "==update state=="
        agent_lt.update_state(agent_gs, ipv_update_method)
        agent_gs.update_state(agent_lt, ipv_update_method)
        # print("estimated gs ipv:", agent_lt.estimated_inter_agent.ipv)
        # print("estimated lt ipv:", agent_gs.estimated_inter_agent.ipv)

    "====save data===="
    if save_data_needed:
        filename = output_directory + '/data/agents_info' \
                   + '_gs_' + str(gs_ipv_sim) \
                   + '_lt_' + str(lt_ipv_sim) + '.pckl'
        f = open(filename, 'wb')
        pickle.dump([agent_lt, agent_gs], f)
        f.close()
        print('gs_' + str(gs_ipv_sim) + '_lt_' + str(lt_ipv_sim), 'saved')

    "====visualization===="
    if final_illustration_needed:
        # get central vertices
        cv_it, _ = get_central_vertices('lt')
        cv_gs, _ = get_central_vertices('gs')

        # set figures
        fig = plt.figure(1)
        plt.title('gs_' + str(gs_ipv_sim) + '_lt_' + str(lt_ipv_sim))
        ax1 = fig.add_subplot(121)
        ax1.set(xlim=[5, 25], ylim=[-15, 15])
        ax2 = fig.add_subplot(122)
        # ax2.set(xlim=[5, 25], ylim=[-15, 15])

        "====show plans at each time step===="
        # central vertices
        ax1.plot(cv_it[:, 0], cv_it[:, 1], 'r-')
        ax1.plot(cv_gs[:, 0], cv_gs[:, 1], 'b-')

        # position at each time step
        ax1.scatter(agent_lt.observed_trajectory[:, 0],
                    agent_lt.observed_trajectory[:, 1],
                    s=100,
                    alpha=0.6,
                    color='red',
                    label='left-turn')
        ax1.scatter(agent_gs.observed_trajectory[:, 0],
                    agent_gs.observed_trajectory[:, 1],
                    s=100,
                    alpha=0.6,
                    color='blue',
                    label='go-straight')

        # full tracks at each time step
        for t in range(num_step):
            lt_track = agent_lt.trj_solution_collection[t]
            ax1.plot(lt_track[:, 0], lt_track[:, 1], '--', color='red')
            gs_track = agent_gs.trj_solution_collection[t]
            ax1.plot(gs_track[:, 0], gs_track[:, 1], '--', color='blue')

        # connect two agents
        for t in range(len(agent_lt.observed_trajectory)):
            ax1.plot([agent_lt.observed_trajectory[t, 0], agent_gs.observed_trajectory[t, 0]],
                     [agent_lt.observed_trajectory[t, 1], agent_gs.observed_trajectory[t, 1]],
                     color='black',
                     alpha=0.2)

        "====show IPV and uncertainty===="
        x_range = np.array(range(len(agent_lt.estimated_inter_agent.ipv_collection)))
        y_lt = np.array(agent_gs.estimated_inter_agent.ipv_collection)
        y_gs = np.array(agent_lt.estimated_inter_agent.ipv_collection)

        # actual ipv
        ax2.plot(x_range, lt_ipv_sim * math.pi / 8 * np.ones_like(x_range),
                 color='red',
                 linewidth=5,
                 label='actual lt IPV')
        ax2.plot(x_range, gs_ipv_sim * math.pi / 8 * np.ones_like(x_range),
                 color='blue',
                 linewidth=5,
                 label='actual gs IPV')

        # estimated ipv
        ax2.plot(x_range, y_lt, color='red', label='estimated lt IPV')
        ax2.plot(x_range, y_gs, color='blue', label='estimated gs IPV')

        # error bar
        if ipv_update_method == 1:
            y_error_lt = np.array(agent_lt.estimated_inter_agent.ipv_error_collection)
            ax2.fill_between(x_range, y_lt - y_error_lt, y_lt + y_error_lt,
                             alpha=0.3,
                             color='red',
                             label='gs IPV error')
            y_error_gs = np.array(agent_gs.estimated_inter_agent.ipv_error_collection)
            ax2.fill_between(x_range, y_gs - y_error_gs, y_gs + y_error_gs,
                             alpha=0.3,
                             color='blue',
                             label='lt IPV error')

        ax1.legend()
        ax2.legend()
        plt.show()


def multi_simulate(process_id, gs_ipv_set, lt_ipv_set):
    count = 0
    num_all = len(gs_ipv_set) * len(lt_ipv_set)
    for gs_ipv_temp in gs_ipv_set:
        for lt_ipv_temp in lt_ipv_set:
            simulate(gs_ipv_temp, lt_ipv_temp)
            count += 1
            print('#======process ', process_id, ':', count / num_all * 100, '%')


if __name__ == '__main__':
    from multiprocessing import Process

    tic = time.perf_counter()

    "multi process"
    # lt_ipv_set_full = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    # processes = [Process(target=multi_simulate, args=(1, [-4], lt_ipv_set_full)),
    #              Process(target=multi_simulate, args=(2, [-3], lt_ipv_set_full)),
    #              Process(target=multi_simulate, args=(3, [-2], lt_ipv_set_full)),
    #              Process(target=multi_simulate, args=(4, [-1], lt_ipv_set_full)),
    #              Process(target=multi_simulate, args=(5, [0], lt_ipv_set_full)),
    #              Process(target=multi_simulate, args=(6, [1], lt_ipv_set_full)),
    #              Process(target=multi_simulate, args=(7, [2], lt_ipv_set_full)),
    #              Process(target=multi_simulate, args=(8, [3], lt_ipv_set_full)),
    #              Process(target=multi_simulate, args=(9, [4], lt_ipv_set_full)),
    #              ]
    # [p.start() for p in processes]  # 开启进程
    # [p.join() for p in processes]  # 等待进程依次结束

    "multi process for used set"
    # processes = [Process(target=multi_simulate, args=(1, [-2], [-2])),
    #              Process(target=multi_simulate, args=(2, [-2], [0])),
    #              Process(target=multi_simulate, args=(3, [-2], [2])),
    #              Process(target=multi_simulate, args=(4, [0], [-2])),
    #              Process(target=multi_simulate, args=(5, [0], [0])),
    #              Process(target=multi_simulate, args=(6, [0], [2])),
    #              Process(target=multi_simulate, args=(7, [2], [-2])),
    #              Process(target=multi_simulate, args=(8, [2], [0])),
    #              Process(target=multi_simulate, args=(9, [2], [2])),
    #              ]
    # [p.start() for p in processes]  # 开启进程
    # [p.join() for p in processes]  # 等待进程依次结束

    "multi process for used set for cooperativeness analysis"
    # processes = [Process(target=multi_simulate, args=(1, [2], [-3])),
    #              Process(target=multi_simulate, args=(2, [2], [-2])),
    #              Process(target=multi_simulate, args=(3, [2], [-1])),
    #              Process(target=multi_simulate, args=(4, [2], [0])),
    #              Process(target=multi_simulate, args=(5, [2], [1])),
    #              Process(target=multi_simulate, args=(6, [2], [2])),
    #              Process(target=multi_simulate, args=(7, [2], [3]))
    #              ]
    # processes = [Process(target=multi_simulate, args=(1, [-3], [2])),
    #              Process(target=multi_simulate, args=(2, [-2], [2])),
    #              Process(target=multi_simulate, args=(3, [-1], [2])),
    #              Process(target=multi_simulate, args=(4, [0], [2])),
    #              Process(target=multi_simulate, args=(5, [1], [2])),
    #              Process(target=multi_simulate, args=(6, [2], [2])),
    #              Process(target=multi_simulate, args=(7, [3], [2]))
    #              ]
    # [p.start() for p in processes]  # 开启进程
    # [p.join() for p in processes]  # 等待进程依次结束

    "single test"
    # for gs_ipv in [1]:
    #     for lt_ipv in [-1]:
    #         print('gs=' + str(gs_ipv), 'lt=' + str(lt_ipv))
    #         simulate(gs_ipv, lt_ipv)
    gs_ipv = 0.06
    lt_ipv = 0.27
    simulate(gs_ipv, lt_ipv)

    toc = time.perf_counter()
    print(f"whole process takes {toc - tic:0.4f} seconds")
