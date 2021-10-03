import scipy.io
import math
import numpy as np
from matplotlib import pyplot as plt
from agent import Agent
from tools.utility import get_central_vertices, smooth_cv

mat = scipy.io.loadmat('./data/NDS_data.mat')
inter_num = mat['interactagentnumpost']
inter_info = mat['interactinfo']

virtual_agent_IPV_range = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]) * math.pi / 9


def analyze_nds(case_id):
    case_info = inter_info[case_id]
    lt_info = case_info[0]

    # find co-present gs agents (not interacting)
    gs_info_multi = case_info[1:inter_num[0, case_id] + 1]

    # find interacting gs agent
    init_id = 0
    inter_o = np.zeros(np.size(gs_info_multi, 0))
    inter_d = np.zeros(np.size(gs_info_multi, 0))
    for i in range(np.size(gs_info_multi, 0)):
        gs_agent_temp = gs_info_multi[i]
        solid_frame = np.nonzero(gs_agent_temp[:, 0])[0]
        solid_range = range(solid_frame[0], solid_frame[-1])
        inter_frame = solid_frame[0] + np.array(
            np.where(gs_agent_temp[solid_range, 1] - lt_info[solid_range, 1] < 0)[0])

        # find start and end frame with each gs agent
        if inter_frame.size > 0:
            if i == init_id:
                inter_o[i] = inter_frame[0]
            else:
                inter_o[i] = max(inter_frame[0], inter_d[i - 1])
            inter_d[i] = max(inter_frame[-1], inter_d[i - 1])
        else:
            init_id += 1

    # identify IPV
    start_time = 0
    ipv_collection = np.zeros_like(lt_info[:, 0:2])
    ipv_error_collection = np.ones_like(lt_info[:, 0:2])
    for t in range(np.size(lt_info, 0)):
        inter_id = None
        for i in range(np.size(gs_info_multi, 0)):
            if inter_o[i] <= t < inter_d[i]:
                inter_id = i
                print('inter_id', inter_id)
                start_time = max(int(inter_o[inter_id]), t - 6)
        # print(start_time)

        if (inter_id is not None) and (t - start_time >= 3):
            # generate two agents
            init_position_lt = lt_info[start_time, 0:2]
            init_velocity_lt = lt_info[start_time, 2:4]
            init_heading_lt = lt_info[start_time, 4]
            agent_lt = Agent(init_position_lt, init_velocity_lt, init_heading_lt, 'lt_nds')
            lt_track = lt_info[start_time:t + 1, 0:2]

            init_position_gs = gs_info_multi[inter_id][start_time, 0:2]
            init_velocity_gs = gs_info_multi[inter_id][start_time, 2:4]
            init_heading_gs = gs_info_multi[inter_id][start_time, 4]
            agent_gs = Agent(init_position_gs, init_velocity_gs, init_heading_gs, 'gs_nds')
            gs_track = gs_info_multi[inter_id][start_time:t + 1, 0:2]

            # estimate ipv
            agent_lt.estimate_self_ipv_in_NDS(lt_track, gs_track)
            ipv_collection[t, 0] = agent_lt.ipv
            ipv_error_collection[t, 0] = agent_lt.ipv_error
            print('left turn', agent_lt.ipv, agent_lt.ipv_error)

            agent_gs.estimate_self_ipv_in_NDS(gs_track, lt_track)
            ipv_collection[t, 1] = agent_gs.ipv
            ipv_error_collection[t, 1] = agent_gs.ipv_error
            print('go straight', agent_gs.ipv, agent_gs.ipv_error)

        elif inter_id is None:
            print('no interaction')
        elif t - start_time < 3:
            print('no results, more observation needed')

        if t > 3:
            x_range = range(max(0, t - 10), t)

            plt.figure(1)
            plt.clf()
            # print(ipv_collection[x_range, 0])
            smoothed_ipv_lt, _ = smooth_cv(np.array([x_range, ipv_collection[x_range, 0]]).T)
            smoothed_ipv_error_lt, _ = smooth_cv(np.array([x_range, ipv_error_collection[x_range, 0]]).T)
            smoothed_x = smoothed_ipv_lt[:, 0]
            plt.plot(smoothed_x, smoothed_ipv_lt[:, 1], 'blue')
            plt.fill_between(smoothed_x, smoothed_ipv_lt[:, 1] - smoothed_ipv_error_lt[:, 1],
                             smoothed_ipv_lt[:, 1] + smoothed_ipv_error_lt[:, 1],
                             alpha=0.4,
                             color='blue',
                             label='estimated lt IPV')

            smoothed_ipv_gs, _ = smooth_cv(np.array([x_range, ipv_collection[x_range, 1]]).T)
            smoothed_ipv_error_gs, _ = smooth_cv(np.array([x_range, ipv_error_collection[x_range, 1]]).T)
            plt.plot(smoothed_x, smoothed_ipv_gs[:, 1], 'red')
            plt.fill_between(smoothed_x, smoothed_ipv_gs[:, 1] - smoothed_ipv_error_gs[:, 1],
                             smoothed_ipv_gs[:, 1] + smoothed_ipv_error_gs[:, 1],
                             alpha=0.4,
                             color='red',
                             label='estimated gs IPV')
            plt.figure(2)
            # print(init_position_lt[0])
            # print(init_position_lt[1])
            plt.scatter(lt_info[t, 0], lt_info[t, 1],
                        s=100,
                        alpha=0.6,
                        color='blue',
                        label='go-straight')
            plt.scatter(gs_info_multi[inter_id][t, 0], gs_info_multi[inter_id][t, 1],
                        s=100,
                        alpha=0.6,
                        color='red',
                        label='go-straight')
            plt.pause(0.3)


if __name__ == '__main__':
    analyze_nds(4)
