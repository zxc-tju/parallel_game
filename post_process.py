import pickle
import math
import gc
import pandas as pd
from matplotlib import pyplot as plt
from tools.utility import get_central_vertices, smooth_cv
import numpy as np

ipv_update_method = 1
show_gif = 0
save_fig = 1
save_data = 0


def show_results(gs_ipv, lt_ipv):

    # import data
    version_num = '19'
    filename = './outputs/version' + str(version_num) + '/data/agents_info' \
               + '_gs_' + str(gs_ipv) \
               + '_lt_' + str(lt_ipv) \
               + '.pckl'
    f = open(filename, 'rb')
    agent_lt, agent_gs = pickle.load(f)
    f.close()

    "save data to excel"
    if save_data:
        lt_ob_trj = agent_lt.observed_trajectory[:, 0:2]
        df_lt_ob_trj = pd.DataFrame(lt_ob_trj)

        lt_trj_coll = agent_lt.trj_solution_collection[0][:, 0:2]
        for i in range(9):
            coll_temp = agent_lt.trj_solution_collection[(i+1)*2][:, 0:2]
            lt_trj_coll = np.concatenate([lt_trj_coll, coll_temp], axis=1)
        df_lt_trj_coll = pd.DataFrame(lt_trj_coll)

        gs_ob_trj = agent_gs.observed_trajectory[:, 0:2]
        df_gs_ob_trj = pd.DataFrame(gs_ob_trj)

        gs_trj_coll = agent_gs.trj_solution_collection[0][:, 0:2]
        for i in range(9):
            coll_temp = agent_gs.trj_solution_collection[(i+1)*2][:, 0:2]
            gs_trj_coll = np.concatenate([gs_trj_coll, coll_temp], axis=1)
        df_gs_trj_coll = pd.DataFrame(gs_trj_coll)

        link = np.concatenate([[lt_ob_trj[0, :]], [gs_ob_trj[0, :]]])
        for i in range(10):
            link = np.concatenate([link,
                                   np.concatenate([[lt_ob_trj[(i+1)*2, :]], [gs_ob_trj[(i+1)*2, :]]])], axis=1)
        df_link = pd.DataFrame(link)

        with pd.ExcelWriter('outputs/excel/' + str(version_num) + '/output'
                            + '_gs_' + str(gs_ipv)
                            + '_lt_' + str(lt_ipv) + '.xlsx') as writer:
            df_lt_ob_trj.to_excel(writer, index=False, sheet_name='lt_ob_trj')
            df_lt_trj_coll.to_excel(writer, index=False, sheet_name='lt_trj_coll')
            df_gs_ob_trj.to_excel(writer, index=False, sheet_name='gs_ob_trj')
            df_gs_trj_coll.to_excel(writer, index=False, sheet_name='gs_trj_coll')
            df_link.to_excel(writer, index=False, sheet_name='link')

    # process data for figures
    cv_init_it, _ = get_central_vertices('lt', None)
    cv_init_gs, _ = get_central_vertices('gs', None)

    if hasattr(agent_lt, 'trajectory'):
        agent_lt_observed_trajectory = agent_lt.trajectory
        agent_gs_observed_trajectory = agent_gs.trajectory
    else:
        agent_lt_observed_trajectory = agent_lt.observed_trajectory
        agent_gs_observed_trajectory = agent_gs.observed_trajectory

    "====final observed_trajectory===="
    if show_gif:
        fig = plt.figure(figsize=(12, 28))  # for showing gif
    else:
        fig = plt.figure(dpi=300, figsize=(12, 18))  # for printing figure
    ax1 = plt.subplot(211)
    ax2 = fig.add_subplot(212)
    img = plt.imread('background_pic/T_intersection.jpg')

    num_frame = len(agent_lt_observed_trajectory)
    for t in range(num_frame):
        ax1.cla()
        ax1.imshow(img, extent=[-9.1, 24.9, -13, 8])
        ax1.set(xlim=[-9.1, 24.9], ylim=[-13, 8])
        # central vertices
        ax1.plot(cv_init_it[:, 0], cv_init_it[:, 1], 'r-')
        ax1.plot(cv_init_gs[:, 0], cv_init_gs[:, 1], 'b-')
        # left-turn
        ax1.scatter(agent_lt_observed_trajectory[:t + 1, 0],
                    agent_lt_observed_trajectory[:t + 1, 1],
                    s=120,
                    alpha=0.4,
                    color='red',
                    label='left-turn')
        # go-straight
        ax1.scatter(agent_gs_observed_trajectory[:t + 1, 0],
                    agent_gs_observed_trajectory[:t + 1, 1],
                    s=120,
                    alpha=0.4,
                    color='blue',
                    label='go-straight')
        if t < len(agent_lt_observed_trajectory) - 1:
            # real-time virtual plans of ## ego ## at time step t
            lt_track = agent_lt.trj_solution_collection[t]
            ax1.plot(lt_track[:, 0], lt_track[:, 1], '--', linewidth=3)
            gs_track = agent_gs.trj_solution_collection[t]
            ax1.plot(gs_track[:, 0], gs_track[:, 1], '--', linewidth=3)
            if ipv_update_method == 1:
                # real-time virtual plans of ## interacting agent ## at time step t
                candidates_lt = agent_lt.estimated_inter_agent.virtual_track_collection[t]
                candidates_gs = agent_gs.estimated_inter_agent.virtual_track_collection[t]
                for track_lt in candidates_lt:
                    ax1.plot(track_lt[:, 0], track_lt[:, 1], color='green', alpha=0.5)
                for track_gs in candidates_gs:
                    ax1.plot(track_gs[:, 0], track_gs[:, 1], color='green', alpha=0.5)
        # position link
        ax1.plot([agent_lt_observed_trajectory[t, 0], agent_gs_observed_trajectory[t, 0]],
                 [agent_lt_observed_trajectory[t, 1], agent_gs_observed_trajectory[t, 1]],
                 color='gray',
                 alpha=0.2)
        if show_gif:
            plt.pause(0.3)
    # full position link
    for t in range(len(agent_lt_observed_trajectory)):
        ax1.plot([agent_lt_observed_trajectory[t, 0], agent_gs_observed_trajectory[t, 0]],
                 [agent_lt_observed_trajectory[t, 1], agent_gs_observed_trajectory[t, 1]],
                 color='gray',
                 alpha=0.1)
    ax1.set_title('gs_' + str(gs_ipv) + '_lt_' + str(lt_ipv), fontsize=12)

    "====ipv estimation===="
    x_range = np.array(range(len(agent_lt.estimated_inter_agent.ipv_collection)))

    # ====draw left turn
    y_lt = np.array(agent_lt.estimated_inter_agent.ipv_collection)
    point_smoothed_lt, _ = smooth_cv(np.array([x_range, y_lt]).T)
    x_smoothed_lt = point_smoothed_lt[:, 0]
    y_lt_smoothed = point_smoothed_lt[:, 1]
    ax2.plot(x_smoothed_lt, y_lt_smoothed,
             alpha=1,
             color='blue',
             label='estimated gs IPV')
    if ipv_update_method == 1:
        # error bar
        y_error_lt = np.array(agent_lt.estimated_inter_agent.ipv_error_collection)
        error_smoothed_lt, _ = smooth_cv(np.array([x_range, y_error_lt]).T)
        y_error_lt_smoothed = error_smoothed_lt[:, 1]
        plt.fill_between(x_smoothed_lt, y_lt_smoothed - y_error_lt_smoothed, y_lt_smoothed + y_error_lt_smoothed,
                         alpha=0.3,
                         color='blue',
                         label='estimated gs IPV')
    # ground truth
    ax2.plot(x_range, gs_ipv * math.pi / 8 * np.ones_like(x_range),
             linewidth=5,
             label='actual gs IPV')

    # ====draw go straight
    y_gs = np.array(agent_gs.estimated_inter_agent.ipv_collection)
    # smoothen data
    point_smoothed_gs, _ = smooth_cv(np.array([x_range, y_gs]).T)
    x_smoothed_gs = point_smoothed_gs[:, 0]
    y_gs_smoothed = point_smoothed_gs[:, 1]
    ax2.plot(x_smoothed_gs, y_gs_smoothed,
             alpha=1,
             color='red',
             label='estimated gs IPV')
    if ipv_update_method == 1:
        # error bar
        y_error_gs = np.array(agent_gs.estimated_inter_agent.ipv_error_collection)
        error_smoothed_gs, _ = smooth_cv(np.array([x_range, y_error_gs]).T)
        y_error_gs_smoothed = error_smoothed_gs[:, 1]
        ax2.fill_between(x_smoothed_gs, y_gs_smoothed - y_error_gs_smoothed, y_gs_smoothed + y_error_gs_smoothed,
                         alpha=0.3,
                         color='red',
                         label='estimated lt IPV')
    # ground truth
    ax2.plot(x_range, lt_ipv * math.pi / 8 * np.ones_like(x_range),
             linewidth=5,
             label='actual lt IPV')

    # save figure
    if save_fig:
        plt.savefig('./outputs/version' + str(version_num) + '/figures/'
                    + 'gs=' + str(gs_ipv)
                    + '_lt=' + str(lt_ipv) + '.png')
        plt.clf()
        plt.close()
        gc.collect()


if __name__ == '__main__':
    # ipv_list = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    ipv_list = [-2, 0, 2]
    # ipv_list = [1]
    for gs in ipv_list:
        for lt in ipv_list:
            show_results(gs, lt)
