import scipy.io
import math
import numpy as np
from matplotlib import pyplot as plt
from agent import Agent, cal_interior_cost, cal_group_cost
from tools.utility import get_central_vertices, smooth_cv

illustration_needed = True

# load data
mat = scipy.io.loadmat('./data/NDS_data.mat')
# full interaction information
inter_info = mat['interaction_info']
'''
inter_info:
0-1: [position x] [position y]
2: [acceleration]
3-5: [velocity x] [velocity y] [velocity overall = sqrt(vx^2+xy^2)]
6: [curvature] (only for left-turn vehicles)
dt = 0.12s 
'''
# the number of go-straight vehicles that interact with the left-turn vehicle
inter_num = mat['interact_agent_num']

virtual_agent_IPV_range = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]) * math.pi / 9


def visualize_nds(case_id):
    # abstract interaction info. of a given case
    case_info = inter_info[case_id]
    # left-turn vehicle
    lt_info = case_info[0]
    # go-straight vehicles
    gs_info_multi = case_info[1:inter_num[0, case_id] + 1]

    fig = plt.figure(1)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax2.set(xlim=[-22, 53], ylim=[-31, 57])
    img = plt.imread('background_pic/Jianhexianxia.jpg')
    ax2.imshow(img, extent=[-22, 53, -31, 57])

    for t in range(np.size(lt_info, 0)):
        t_end = t + 6
        ax1.cla()
        ax1.set(xlim=[-22, 53], ylim=[-31, 57])
        img = plt.imread('background_pic/Jianhexianxia.jpg')
        ax1.imshow(img, extent=[-22, 53, -31, 57])

        # position of go-straight vehicles
        for gs_id in range(np.size(gs_info_multi, 0)):
            if np.size(gs_info_multi[gs_id], 0) > t and not gs_info_multi[gs_id][t, 0] == 0:
                # position
                ax1.scatter(gs_info_multi[gs_id][t, 0], gs_info_multi[gs_id][t, 1],
                            s=80,
                            alpha=0.3,
                            color='red',
                            label='go-straight')
                # future track
                t_end = min(t + 6, np.size(gs_info_multi[gs_id], 0))
                ax1.plot(gs_info_multi[gs_id][t:t_end, 0], gs_info_multi[gs_id][t:t_end, 1],
                         alpha=0.8,
                         color='red')

        # position of left-turn vehicle
        ax1.scatter(lt_info[t, 0], lt_info[t, 1],
                    s=80,
                    alpha=0.3,
                    color='blue',
                    label='left-turn')
        # future track
        ax1.plot(lt_info[t:t_end, 0], lt_info[t:t_end, 1],
                 alpha=0.8,
                 color='blue')
        # ax1.legend()
        plt.pause(0.1)

    # show full track of all agents
    ax2.plot(lt_info[:, 0], lt_info[:, 1],
             alpha=0.8,
             color='blue')
    for gs_id in range(np.size(gs_info_multi, 0)):
        # find solid frames
        frames = np.where(gs_info_multi[gs_id][:, 0] < 1e-3)
        # the first solid frame id
        frame_start = len(frames[0])
        # tracks
        ax2.plot(gs_info_multi[gs_id][frame_start:, 0], gs_info_multi[gs_id][frame_start:, 1],
                 alpha=0.8,
                 color='red')
    plt.show()


def analyze_nds(case_id):
    case_info = inter_info[case_id]
    lt_info = case_info[0]

    # find co-present gs agents (not necessarily interacting)
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

    # set figure
    fig = plt.figure(1)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for t in range(np.size(lt_info, 0)):

        "find current interacting agent"
        inter_id = None
        for i in range(np.size(gs_info_multi, 0)):
            if inter_o[i] <= t < inter_d[i]:
                inter_id = i
                print('inter_id', inter_id)
                start_time = max(int(inter_o[inter_id]), t - 10)

        "IPV estimation process"
        if (inter_id is not None) and (t - start_time > 3):

            "====simulation-based method===="
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
            "====end of simulation-based method===="

            "====cost-based method===="
            # # load observed trajectories
            # lt_track_observed = lt_info[start_time:t + 1, 0:2]
            # gs_track_observed = gs_info_multi[inter_id][start_time:t + 1, 0:2]
            #
            # # generate two agents
            # init_position_lt = lt_info[start_time, 0:2]
            # init_velocity_lt = lt_info[start_time, 2:4]
            # init_heading_lt = lt_info[start_time, 4]
            # agent_lt = Agent(init_position_lt, init_velocity_lt, init_heading_lt, 'lt_nds')
            # agent_lt.ipv = 0
            # init_position_gs = gs_info_multi[inter_id][start_time, 0:2]
            # init_velocity_gs = gs_info_multi[inter_id][start_time, 2:4]
            # init_heading_gs = gs_info_multi[inter_id][start_time, 4]
            # agent_gs = Agent(init_position_gs, init_velocity_gs, init_heading_gs, 'gs_nds')
            # agent_gs.ipv = 0
            #
            # # plan under selfish assumption
            # lt_track_selfish = agent_lt.solve_game_IBR(gs_track_observed)
            # lt_track_selfish = lt_track_selfish[:, 0:2]
            # gs_track_selfish = agent_lt.solve_game_IBR(lt_track_observed)
            # gs_track_selfish = gs_track_selfish[:, 0:2]
            #
            # # cost results in observation
            # lt_interior_cost_observed = cal_interior_cost([], lt_track_observed, 'lt_nds')
            # gs_interior_cost_observed = cal_interior_cost([], gs_track_observed, 'gs_nds')
            # group_cost_observed = cal_group_cost([lt_track_observed, gs_track_observed])
            #
            # # cost result in assumption
            # lt_interior_cost_assumed = cal_interior_cost([], lt_track_selfish, 'lt_nds')
            # gs_interior_cost_assumed = cal_interior_cost([], gs_track_selfish, 'gs_nds')
            # group_cost_lt_assumed = cal_group_cost([lt_track_selfish, gs_track_observed])
            # group_cost_gs_assumed = cal_group_cost([lt_track_observed, gs_track_selfish])

            # ipv_collection[t, 0] =
            # ipv_collection[t, 1] =

            "====end of cost-based method===="

            "illustration"
            if illustration_needed:
                ax1.cla()
                ax1.set(ylim=[-2, 2])

                x_range = range(max(0, t - 10), t)
                smoothed_ipv_lt, _ = smooth_cv(np.array([x_range, ipv_collection[x_range, 0]]).T)
                smoothed_ipv_error_lt, _ = smooth_cv(np.array([x_range, ipv_error_collection[x_range, 0]]).T)
                smoothed_x = smoothed_ipv_lt[:, 0]
                # plot ipv
                ax1.plot(smoothed_x, smoothed_ipv_lt[:, 1], 'blue')
                # plot error bar
                ax1.fill_between(smoothed_x, smoothed_ipv_lt[:, 1] - smoothed_ipv_error_lt[:, 1],
                                 smoothed_ipv_lt[:, 1] + smoothed_ipv_error_lt[:, 1],
                                 alpha=0.4,
                                 color='blue',
                                 label='estimated lt IPV')

                smoothed_ipv_gs, _ = smooth_cv(np.array([x_range, ipv_collection[x_range, 1]]).T)
                smoothed_ipv_error_gs, _ = smooth_cv(np.array([x_range, ipv_error_collection[x_range, 1]]).T)
                # plot ipv
                ax1.plot(smoothed_x, smoothed_ipv_gs[:, 1], 'red')
                # plot error bar
                ax1.fill_between(smoothed_x, smoothed_ipv_gs[:, 1] - smoothed_ipv_error_gs[:, 1],
                                 smoothed_ipv_gs[:, 1] + smoothed_ipv_error_gs[:, 1],
                                 alpha=0.4,
                                 color='red',
                                 label='estimated gs IPV')
                ax1.legend()

                # show trajectory and plans
                ax2.cla()
                ax2.set(xlim=[-22, 53], ylim=[-31, 57])
                img = plt.imread('background_pic/Jianhexianxia.jpg')
                ax2.imshow(img, extent=[-22, 53, -31, 57])
                cv1, _ = get_central_vertices('lt_nds', [lt_info[start_time, 0], lt_info[start_time, 1]])
                cv2, _ = get_central_vertices('gs_nds', [gs_info_multi[inter_id][start_time, 0],
                                                         gs_info_multi[inter_id][start_time, 1]])
                ax2.plot(cv1[:, 0], cv1[:, 1])
                ax2.plot(cv2[:, 0], cv2[:, 1])

                # actual track
                ax2.scatter(lt_info[start_time:t, 0], lt_info[start_time:t, 1],
                            s=50,
                            alpha=0.5,
                            color='blue',
                            label='left-turn')
                candidates_lt = agent_lt.virtual_track_collection
                for track_lt in candidates_lt:
                    ax2.plot(track_lt[:, 0], track_lt[:, 1], color='green', alpha=0.5)
                ax2.scatter(gs_info_multi[inter_id][start_time:t, 0], gs_info_multi[inter_id][start_time:t, 1],
                            s=50,
                            alpha=0.5,
                            color='red',
                            label='go-straight')
                candidates_gs = agent_gs.virtual_track_collection
                for track_gs in candidates_gs:
                    ax2.plot(track_gs[:, 0], track_gs[:, 1], color='green', alpha=0.5)
                ax2.legend()

                plt.pause(0.3)

        elif inter_id is None:
            print('no interaction')

        elif t - start_time < 3:
            print('no results, more observation needed')


if __name__ == '__main__':
    nds_case_id = 15

    "analyze IPV in NDS"
    # analyze_nds(nds_case_id)

    "show trajectories in NDS"
    visualize_nds(nds_case_id)
