import scipy.io
import math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mt
from agent import Agent, cal_interior_cost, cal_group_cost
from tools.utility import get_central_vertices, smooth_cv
import pandas as pd
from scipy.interpolate import interp2d

from openpyxl import load_workbook

illustration_needed = False
print_needed = False

# load data
mat = scipy.io.loadmat('./data/NDS_data.mat')
# full interaction information
inter_info = mat['interaction_info']
'''
inter_info:
0-1: [position x] [position y]
2-3: [velocity x] [velocity y]
4: [heading]
5: [velocity overall = sqrt(vx^2+xy^2)]
6: [curvature] (only for left-turn vehicles)
dt = 0.12s 
'''
# the number of go-straight vehicles that interact with the left-turn vehicle
inter_num = mat['interact_agent_num']

virtual_agent_IPV_range = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]) * math.pi / 9


def draw_rectangle(x, y, deg):
    car_len = 5
    car_wid = 2
    fig = plt.figure()
    ax = fig.add_subplot(111)

    r1 = patches.Rectangle((x - car_wid / 2, y - car_len / 2), car_wid, car_len, color="blue", alpha=0.50)
    r2 = patches.Rectangle((x - car_wid / 2, y - car_len / 2), car_wid, car_len, color="red", alpha=0.50)

    t2 = mt.Affine2D().rotate_deg_around(x, y, deg) + ax.transData
    r2.set_transform(t2)

    ax.add_patch(r1)
    ax.add_patch(r2)

    plt.grid(True)
    plt.axis('equal')

    plt.show()


def visualize_nds(case_id):
    # abstract interaction info. of a given case
    case_info = inter_info[case_id]
    # left-turn vehicle
    lt_info = case_info[0]
    # go-straight vehicles
    gs_info_multi = case_info[1:inter_num[0, case_id] + 1]

    fig = plt.figure(1)
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax2.set(xlim=[-22, 53], ylim=[-31, 57])
    img = plt.imread('background_pic/Jianhexianxia.jpg')
    ax2.imshow(img, extent=[-22, 53, -31, 57])

    for t in range(np.size(lt_info, 0)):
        if t in {2, 14, 30, 40}:
            print('pause!')
        t_end = t + 10
        ax1.cla()
        ax1.set(xlim=[-22, 53], ylim=[-31, 57])
        img = plt.imread('background_pic/Jianhexianxia.jpg')
        ax1.imshow(img, extent=[-22, 53, -31, 57])

        # position of go-straight vehicles
        for gs_id in range(np.size(gs_info_multi, 0)):
            if np.size(gs_info_multi[gs_id], 0) > t and not gs_info_multi[gs_id][t, 0] == 0:
                # position
                ax1.scatter(gs_info_multi[gs_id][t, 0], gs_info_multi[gs_id][t, 1],
                            s=120,
                            alpha=0.9,
                            color='red',
                            label='go-straight')
                # future track
                t_end_gs = min(t + 10, np.size(gs_info_multi[gs_id], 0))
                ax1.plot(gs_info_multi[gs_id][t:t_end_gs, 0], gs_info_multi[gs_id][t:t_end_gs, 1],
                         alpha=0.8,
                         color='red')

        # position of left-turn vehicle
        ax1.scatter(lt_info[t, 0], lt_info[t, 1],
                    s=120,
                    alpha=0.9,
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


def find_inter_od(case_id):
    """
    find the starting and end frame of each FC agent that interacts with LT agent
    :param case_id:
    :return:
    """
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
        if inter_frame.size > 1:
            if i == init_id:
                inter_o[i] = inter_frame[0]
            else:
                inter_o[i] = max(inter_frame[0], inter_d[i - 1])
            inter_d[i] = max(inter_frame[-1], inter_d[i - 1])
        else:
            init_id += 1
    return inter_o, inter_d


def analyze_nds(case_id):
    """
    estimate IPV in natural driving data and write results into excels
    :param case_id:
    :return:
    """
    inter_o, inter_d = find_inter_od(case_id)
    case_info = inter_info[case_id]
    lt_info = case_info[0]

    # find co-present gs agents (not necessarily interacting)
    gs_info_multi = case_info[1:inter_num[0, case_id] + 1]

    # initialize IPV
    start_time = 0
    ipv_collection = np.zeros_like(lt_info[:, 0:2])
    ipv_error_collection = np.ones_like(lt_info[:, 0:2])

    # set figure
    if illustration_needed:
        fig = plt.figure(1)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

    inter_id = 0
    inter_id_save = inter_id
    file_name = './outputs/NDS_analysis/v3/' + str(case_id) + '.xlsx'

    for t in range(np.size(lt_info, 0)):

        "find current interacting agent"
        flag = 0
        for i in range(np.size(gs_info_multi, 0)):
            if inter_o[i] <= t < inter_d[i]:  # switch to next interacting agent
                # update interaction info
                flag = 1
                inter_id = i
                if print_needed:
                    print('inter_id', inter_id)
                start_time = max(int(inter_o[inter_id]), t - 10)

        # save data of last one
        if inter_id_save < inter_id or t == inter_d[-1]:
            # if inter_d[inter_id_save] - inter_o[inter_id_save] > 3:
            '''
            inter_id_save < inter_id：  interacting agent changed
            t == inter_d[-1]:  end frame of the last agent
            inter_d[inter_id_save]-inter_o[inter_id_save] > 3：  interacting period is long enough
            '''
            # save data into an excel with the format of:
            # 0-ipv_lt | ipv_lt_error | lt_px | lt_py  | lt_vx  | lt_vy  | lt_heading  |...
            # 7-ipv_gs | ipv_gs_error | gs_px | gs_py  | gs_vx  | gs_vy  | gs_heading  |

            df_ipv_lt = pd.DataFrame(ipv_collection[int(inter_o[inter_id_save]): int(inter_d[inter_id_save]), 0],
                                     columns=["ipv_lt"])
            df_ipv_lt_error = pd.DataFrame(
                ipv_error_collection[int(inter_o[inter_id_save]): int(inter_d[inter_id_save]), 0],
                columns=["ipv_lt_error"])
            df_motion_lt = pd.DataFrame(lt_info[int(inter_o[inter_id_save]): int(inter_d[inter_id_save]), 0:5],
                                        columns=["lt_px", "lt_py", "lt_vx", "lt_vy", "lt_heading"])

            df_ipv_gs = pd.DataFrame(ipv_collection[int(inter_o[inter_id_save]): int(inter_d[inter_id_save]), 1],
                                     columns=["ipv_gs"])
            df_ipv_gs_error = pd.DataFrame(
                ipv_error_collection[int(inter_o[inter_id_save]): int(inter_d[inter_id_save]), 1],
                columns=["ipv_gs_error"])
            df_motion_gs = pd.DataFrame(gs_info_multi[inter_id_save]
                                        [int(inter_o[inter_id_save]): int(inter_d[inter_id_save]), 0:5],
                                        columns=["gs_px", "gs_py", "gs_vx", "gs_vy", "gs_heading"])

            if inter_id_save == 0:
                with pd.ExcelWriter(file_name) as writer:
                    df_ipv_lt.to_excel(writer, startcol=0, index=False, sheet_name=str(inter_id_save))
                    df_ipv_lt_error.to_excel(writer, startcol=1, index=False, sheet_name=str(inter_id_save))
                    df_motion_lt.to_excel(writer, startcol=2, index=False, sheet_name=str(inter_id_save))

                    df_ipv_gs.to_excel(writer, startcol=7, index=False, sheet_name=str(inter_id_save))
                    df_ipv_gs_error.to_excel(writer, startcol=8, index=False, sheet_name=str(inter_id_save))
                    df_motion_gs.to_excel(writer, startcol=9, index=False, sheet_name=str(inter_id_save))
            else:
                with pd.ExcelWriter(file_name, mode="a", if_sheet_exists="overlay") as writer:
                    df_ipv_lt.to_excel(writer, startcol=0, index=False, sheet_name=str(inter_id_save))
                    df_ipv_lt_error.to_excel(writer, startcol=1, index=False, sheet_name=str(inter_id_save))
                    df_motion_lt.to_excel(writer, startcol=2, index=False, sheet_name=str(inter_id_save))

                    df_ipv_gs.to_excel(writer, startcol=7, index=False, sheet_name=str(inter_id_save))
                    df_ipv_gs_error.to_excel(writer, startcol=8, index=False, sheet_name=str(inter_id_save))
                    df_motion_gs.to_excel(writer, startcol=9, index=False, sheet_name=str(inter_id_save))

            inter_id_save = inter_id

        "IPV estimation process"
        if flag and (t - start_time > 3):

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

            agent_gs.estimate_self_ipv_in_NDS(gs_track, lt_track)
            ipv_collection[t, 1] = agent_gs.ipv
            ipv_error_collection[t, 1] = agent_gs.ipv_error

            if print_needed:
                print('left turn', agent_lt.ipv, agent_lt.ipv_error)
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
            if print_needed:
                print('no interaction')

        elif t - start_time < 3:
            if print_needed:
                print('no results, more observation needed')


def analyze_ipv_in_nds(case_id, fig=False):
    file_name = './outputs/NDS_analysis/v3/' + str(case_id) + '.xlsx'
    file = pd.ExcelFile(file_name)
    num_sheet = len(file.sheet_names)
    # print(num_sheet)
    start_x = 0
    crossing_id = -1

    for i in range(num_sheet):
        "get ipv data from excel"
        df_ipv_data = pd.read_excel(file_name, sheet_name=i)
        ipv_data_temp = df_ipv_data.values
        ipv_value_lt, ipv_value_gs = ipv_data_temp[:, 0], ipv_data_temp[:, 7]
        ipv_error_lt, ipv_error_gs = ipv_data_temp[:, 1], ipv_data_temp[:, 8]

        # find cross event
        x_lt = ipv_data_temp[:, 2]
        x_gs = ipv_data_temp[:, 9]
        delta_x = x_lt - x_gs  # x position of LT is larger than that of interacting FC
        if len(delta_x) > 0:
            if np.max(delta_x) > 0 and crossing_id == -1:
                crossing_id = i

        "draw ipv value and error bar"

        if fig:
            x = start_x + np.arange(len(ipv_value_lt))
            start_x = start_x + len(ipv_value_lt)
            print(start_x)

            if len(x) > 6:
                # left turn
                smoothed_ipv_value_lt, _ = smooth_cv(np.array([x, ipv_value_lt]).T)
                smoothed_ipv_error_lt, _ = smooth_cv(np.array([x, ipv_error_lt]).T)
                plt.plot(smoothed_ipv_value_lt[:, 0], smoothed_ipv_value_lt[:, 1],
                         color='blue')
                plt.fill_between(smoothed_ipv_value_lt[:, 0], smoothed_ipv_value_lt[:, 1] - smoothed_ipv_error_lt[:, 1],
                                 smoothed_ipv_value_lt[:, 1] + smoothed_ipv_error_lt[:, 1],
                                 alpha=0.4,
                                 color='blue')

                # go straight
                smoothed_ipv_value_gs, _ = smooth_cv(np.array([x, ipv_value_gs]).T)
                smoothed_ipv_error_gs, _ = smooth_cv(np.array([x, ipv_error_gs]).T)
                plt.plot(smoothed_ipv_value_gs[:, 0], smoothed_ipv_value_gs[:, 1],
                         color='red')
                plt.fill_between(smoothed_ipv_value_gs[:, 0], smoothed_ipv_value_gs[:, 1] - smoothed_ipv_error_gs[:, 1],
                                 smoothed_ipv_value_gs[:, 1] + smoothed_ipv_error_gs[:, 1],
                                 alpha=0.4,
                                 color='red')

            else:  # too short to be fitted
                # left turn
                plt.plot(x, ipv_value_lt,
                         color='red')
                plt.fill_between(x, ipv_value_lt - ipv_error_lt,
                                 ipv_value_lt + ipv_error_lt,
                                 alpha=0.4,
                                 color='red',
                                 label='estimated lt IPV')

                # go straight
                plt.plot(x, ipv_value_gs,
                         color='blue')
                plt.fill_between(x, ipv_value_gs - ipv_error_gs,
                                 ipv_value_gs + ipv_error_gs,
                                 alpha=0.4,
                                 color='blue',
                                 label='estimated gs IPV')
            # plt.pause(1)
    plt.show()

    # save ipv during the crossing event
    case_data_crossing = []
    case_data_non_crossing = []
    if not crossing_id == -1:
        df_data = pd.read_excel(file_name, sheet_name=crossing_id)
        case_data_crossing = df_data.values[4:, :]

    for sheet_id in range(num_sheet):
        if not sheet_id == crossing_id:
            df_data = pd.read_excel(file_name, sheet_name=sheet_id)
            case_data_non_crossing.append(df_data.values[4:, :])

    return crossing_id, case_data_crossing, case_data_non_crossing


def show_ipv_distribution():
    ipv_cross_lt = []
    ipv_cross_gs = []
    ipv_non_cross_lt = []
    ipv_non_cross_gs = []
    for i in range(np.size(inter_info, 0)):

        _, ipv_cross_temp, ipv_non_cross_temp = analyze_ipv_in_nds(i, False)
        if len(ipv_cross_temp) > 0:
            ipv_cross_lt.append(ipv_cross_temp[:, 0])
            ipv_cross_gs.append(ipv_cross_temp[:, 7])
        if len(ipv_non_cross_temp) > 0:
            for idx in range(len(ipv_non_cross_temp)):
                # print(ipv_non_cross[idx][:, 0])
                ipv_non_cross_lt.append(ipv_non_cross_temp[idx][:, 0])
                ipv_non_cross_gs.append(ipv_non_cross_temp[idx][:, 7])

    "calculate mean ipv value of each type"
    mean_ipv_cross_lt = np.array([np.mean(ipv_cross_lt[0])])
    mean_ipv_cross_gs = np.array([np.mean(ipv_cross_gs[0])])
    mean_ipv_non_cross_lt = np.array([np.mean(ipv_non_cross_lt[0])])
    mean_ipv_non_cross_gs = np.array([np.mean(ipv_non_cross_gs[0])])
    for i in range(len(ipv_cross_lt) - 1):
        if np.size(ipv_cross_lt[i + 1], 0) > 4:
            mean_temp1 = np.array([np.mean(ipv_cross_lt[i + 1])])
            mean_ipv_cross_lt = np.concatenate((mean_ipv_cross_lt, mean_temp1), axis=0)
    for i in range(len(ipv_cross_gs) - 1):
        if np.size(ipv_cross_gs[i + 1], 0) > 4:
            mean_temp2 = np.array([np.mean(ipv_cross_gs[i + 1])])
            mean_ipv_cross_gs = np.concatenate((mean_ipv_cross_gs, mean_temp2), axis=0)
    for i in range(len(ipv_non_cross_lt) - 1):
        if np.size(ipv_non_cross_lt[i + 1], 0) > 4:
            mean_temp3 = np.array([np.mean(ipv_non_cross_lt[i + 1])])
            mean_ipv_non_cross_lt = np.concatenate((mean_ipv_non_cross_lt, mean_temp3), axis=0)
    for i in range(len(ipv_non_cross_gs) - 1):
        if np.size(ipv_non_cross_gs[i + 1], 0) > 4:
            mean_temp4 = np.array([np.mean(ipv_non_cross_gs[i + 1])])
            mean_ipv_non_cross_gs = np.concatenate((mean_ipv_non_cross_gs, mean_temp4), axis=0)

    filename = './outputs/ipv_distribution.xlsx'
    with pd.ExcelWriter(filename) as writer:

        data1 = np.vstack((mean_ipv_cross_gs, mean_ipv_cross_lt))
        df_ipv_distribution = pd.DataFrame(data1.T)
        df_ipv_distribution.to_excel(writer, startcol=0, index=False)

        data2 = np.vstack((mean_ipv_non_cross_gs, mean_ipv_non_cross_lt))
        df_ipv_distribution = pd.DataFrame(data2.T)
        df_ipv_distribution.to_excel(writer, startcol=2, index=False)

    plt.figure(1)
    plt.title('Left-turn vehicle rushed')
    plt.hist(mean_ipv_cross_lt,
             alpha=0.5,
             color='blue',
             label='left-turn vehicle')
    plt.hist(mean_ipv_cross_gs,
             alpha=0.5,
             color='red',
             label='go-straight vehicle')
    plt.legend()
    plt.xlabel('IPV')
    plt.ylabel('Counts')

    plt.figure(2)
    plt.title('Left-turn vehicle yielded')
    plt.hist(mean_ipv_non_cross_lt,
             alpha=0.5,
             color='blue',
             label='left-turn vehicle')
    plt.hist(mean_ipv_non_cross_gs,
             alpha=0.5,
             color='red',
             label='go-straight vehicle')
    plt.legend()
    plt.xlabel('IPV')
    plt.ylabel('Counts')
    plt.show()


def cal_pet(trj_a, trj_b):
    min_dis = 99
    min_dis2cv_index = None
    for i in range(np.size(trj_b, 0)):
        dis2cv_lt_temp = np.linalg.norm(trj_a - trj_b, axis=1)
        min_dis2cv_index_temp = np.where(np.amin(dis2cv_lt_temp) == dis2cv_lt_temp)
        min_dis2cv_temp = dis2cv_lt_temp[min_dis2cv_index_temp[0]]
        if min_dis2cv_temp[0] < min_dis:
            min_dis = min_dis2cv_temp[0]
            min_dis2cv_index = min_dis2cv_index_temp[0]
    conflict_point = trj_a[min_dis2cv_index[0], :]

    # find the time step when being near the conflict point
    dis2cv_lt = trj_a - conflict_point
    dis2cv_lt_norm = np.linalg.norm(dis2cv_lt, axis=1)
    dis2cv_lt_min = np.amin(dis2cv_lt_norm)
    t_step_lt = np.where(dis2cv_lt_min == dis2cv_lt_norm)
    t_step_lt = t_step_lt[0][0]
    if trj_a[t_step_lt, 0] > conflict_point[0]:
        t_step_lt = t_step_lt - 1

    dis2cv_gs = trj_b - conflict_point
    dis2cv_gs_norm = np.linalg.norm(dis2cv_gs, axis=1)
    dis2cv_gs_min = np.amin(dis2cv_gs_norm)
    t_step_gs = np.where(dis2cv_gs_min == dis2cv_gs_norm)
    t_step_gs = t_step_gs[0][0]
    if trj_b[t_step_gs, 1] > conflict_point[1]:
        t_step_gs = t_step_gs - 1

    # calculate PET
    pet = np.abs(t_step_gs - t_step_lt) * 0.12 + 0.24
    return pet, t_step_lt, t_step_gs, conflict_point


def divide_pet_in_nds():
    pet_comp = []
    pet_coop = []
    pet_collection = []

    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    ax1.set(xlim=[-23, 53], ylim=[-31, 57])
    img = plt.imread('background_pic/Jianhexianxia.jpg')
    ax1.imshow(img, extent=[-23, 53, -31, 57])

    for case_index in range(131):
        if case_index not in {93, 99, 114}:
            # 93、99：no interaction
            # 114: undivided 2 cases

            cross_id, ipv_data_cross, ipv_data_non_cross = analyze_ipv_in_nds(case_index)
            o, _ = find_inter_od(case_index)
            start_frame = int(o[cross_id])

            # abstract interaction info. of a given case
            case_info = inter_info[case_index]
            # go-straight vehicles
            gs_info_multi = case_info[1:inter_num[0, case_index] + 1]

            if not cross_id == -1 and not case_index == 114:
                gs_trj_case = gs_info_multi[cross_id][start_frame:, 0:2]
                lt_trj_case = inter_info[case_index][0][start_frame:, 0:2]

                pet_temp, t_step_lt, t_step_gs, conflict_point = cal_pet(lt_trj_case, gs_trj_case[i, :])

                lt_ipv = np.mean(ipv_data_cross[:, 0])
                gs_ipv = np.mean(ipv_data_cross[:, 7])

                pet_collection.append([lt_ipv, gs_ipv, pet_temp])

                ax1.scatter(lt_trj_case[t_step_lt, 0], lt_trj_case[t_step_lt, 1])
                ax1.scatter(gs_trj_case[t_step_gs, 0], gs_trj_case[t_step_gs, 1])
                ax1.scatter(conflict_point[0], conflict_point[1], color="black")

                if np.mean(ipv_data_cross[:, 0] * (1 - ipv_data_cross[:, 1])) < 0:  # weighted according to confidence
                    ax1.plot(lt_trj_case[:, 0], lt_trj_case[:, 1], color="red",
                             alpha=-np.mean(ipv_data_cross[:, 0] * (1 - ipv_data_cross[:, 1])) / 1.57)
                    ax1.plot(gs_trj_case[:, 0], gs_trj_case[:, 1], color="red",
                             alpha=-np.mean(ipv_data_cross[:, 0] * (1 - ipv_data_cross[:, 1])) / 1.57)

                    pet_comp.append(pet_temp)

                else:
                    ax1.plot(lt_trj_case[:, 0], lt_trj_case[:, 1], color="green",
                             alpha=np.mean(ipv_data_cross[:, 0] * (1 - ipv_data_cross[:, 1])) / 1.57)
                    ax1.plot(gs_trj_case[:, 0], gs_trj_case[:, 1], color="green",
                             alpha=np.mean(ipv_data_cross[:, 0] * (1 - ipv_data_cross[:, 1])) / 1.57)

                    pet_coop.append(pet_temp)

    plt.figure(2)
    plt.title('PET distribution (grouped)')
    plt.hist(pet_comp,
             alpha=0.5,
             color='red',
             label='competitive')
    plt.hist(pet_coop,
             alpha=0.5,
             color='green',
             label='cooperative')
    plt.legend()
    plt.xlabel('PET')
    plt.ylabel('Counts')

    PET_collection_array = np.array(pet_collection)

    # save pet data
    filename = './outputs/pet_distribution.xlsx'
    with pd.ExcelWriter(filename) as writer:
        df_pet_distribution = pd.DataFrame(PET_collection_array)
        df_pet_distribution.to_excel(writer, startcol=0, index=False)
        df_pet_comp = pd.DataFrame(pet_comp)
        df_pet_coop = pd.DataFrame(pet_coop)
        df_pet_comp.to_excel(writer, startcol=0, index=False, sheet_name="competitive")
        df_pet_coop.to_excel(writer, startcol=0, index=False, sheet_name="cooperative")


def show_crossing_event(case_index):
    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    ax1.set(xlim=[-23, 53], ylim=[-31, 57])
    img = plt.imread('background_pic/Jianhexianxia.jpg')
    ax1.imshow(img, extent=[-23, 53, -31, 57])

    cross_id, data_cross, data_non_cross = analyze_ipv_in_nds(case_index)

    # save data into an excel with the format of:
    # 0-ipv_lt | ipv_lt_error | lt_px | lt_py  | lt_vx  | lt_vy  | lt_heading  |...
    # 7-ipv_gs | ipv_gs_error | gs_px | gs_py  | gs_vx  | gs_vy  | gs_heading  |

    if not cross_id == -1 and not case_index == 114:
        lt_trj_case = data_cross[:, 2:4]
        gs_trj_case = data_cross[:, 9:11]
        ax1.plot(lt_trj_case[:, 0], lt_trj_case[:, 1])
        ax1.plot(gs_trj_case[:, 0], gs_trj_case[:, 1])
        lt_mean_ipv = np.mean(data_cross[4:, 0])
        gs_mean_ipv = np.mean(data_cross[4:, 7])
        ax1.text(0, 60, 'LT:'+str(lt_mean_ipv), fontsize=10)
        ax1.text(0, 65, 'GS:'+str(gs_mean_ipv), fontsize=10)


if __name__ == '__main__':
    "calculate ipv in NDS"
    # estimate IPV in natural driving data and write results into excels (along with all agents' motion info)
    # for case_index in range(131):
    #     analyze_nds(case_index)
    # analyze_nds(30)

    "show trajectories in NDS"
    # visualize_nds(114)

    "find crossing event and the ipv of yield front-coming vehicle (if there is)"
    # cross_id, ipv_data_cross, ipv_data_non_cross = analyze_ipv_in_nds(21, True)

    "show ipv distribution in whole dataset"
    # show_ipv_distribution()

    "find the origin and ending of the each interaction event in a single case"
    # o, d = find_inter_od(30)

    # draw_rectangle(5, 5, 45)

    "divide pet distribution according to the ipv of two agents"
    # divide_pet_in_nds()

    "show crossing trajectories in a case"
    show_crossing_event(30)
