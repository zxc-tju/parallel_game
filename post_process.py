"""
analysis of simulation results
"""

import pickle
import xlsxwriter
import math
import gc
import pandas as pd
from matplotlib import pyplot as plt
from tools.utility import get_central_vertices, smooth_ployline, savitzky_golay
from NDS_analysis import cal_pet
import numpy as np

ipv_update_method = 1
show_gif = 0
save_fig = 0
save_data = 1


def get_results(rd, case_id):
    # import data
    version_num = '27'
    filedir = '../data/3_parallel_game_outputs/simulation/version' + str(version_num)
    filename = filedir + '/data/agents_info' + '_round_' + str(rd) + '_case_' + str(case_id) + '.pckl'
    f = open(filename, 'rb')
    agent_lt, agent_gs = pickle.load(f)
    f.close()

    "====data abstraction===="
    # lt track (observed and planned)
    lt_ob_trj = agent_lt.observed_trajectory[:, 0:2]
    lt_trj_coll = agent_lt.trj_solution_collection[0][:, 0:2]
    # skip one trj every two trjs
    for i in range(int(len(lt_ob_trj) / 2 - 1)):
        coll_temp = agent_lt.trj_solution_collection[(i + 1) * 2][:, 0:2]
        lt_trj_coll = np.concatenate([lt_trj_coll, coll_temp], axis=1)

    # gs track (observed and planned)
    gs_ob_trj = agent_gs.observed_trajectory[:, 0:2]
    gs_trj_coll = agent_gs.trj_solution_collection[0][:, 0:2]
    # skip one trj every two trjs
    for i in range(int(len(lt_ob_trj) / 2 - 1)):
        coll_temp = agent_gs.trj_solution_collection[(i + 1) * 2][:, 0:2]
        gs_trj_coll = np.concatenate([gs_trj_coll, coll_temp], axis=1)

    # link from gs to lt
    link = np.concatenate([[lt_ob_trj[0, :]], [gs_ob_trj[0, :]]])
    # skip one trj every two trjs
    for i in range(int(len(lt_ob_trj) / 2)):
        link = np.concatenate([link, np.concatenate([[lt_ob_trj[(i + 1) * 2, :]],
                                                     [gs_ob_trj[(i + 1) * 2, :]]])], axis=1)
    # process data for figures
    cv_lt, progress_lt = get_central_vertices('lt', None)
    cv_gs, progress_gs = get_central_vertices('gs', None)

    "====calculate PET===="
    pet, _, _ = cal_pet(lt_ob_trj, gs_ob_trj, 'apet')

    # ----archived version---- #
    # # lt speed
    # v_lt = np.linalg.norm(agent_lt.observed_trajectory[:, 2:4], axis=1)
    # v_lt_smoothed = savitzky_golay(v_lt, 5, 3)
    #
    # # gs speed
    # for i in range(int(len(lt_ob_trj) / 2 - 1)):
    #     coll_temp = agent_gs.trj_solution_collection[(i + 1) * 2][:, 0:2]
    #     gs_trj_coll = np.concatenate([gs_trj_coll, coll_temp], axis=1)
    # v_gs = np.linalg.norm(agent_gs.observed_trajectory[:, 2:4], axis=1)
    # v_gs_smoothed = savitzky_golay(v_gs, 5, 3)
    #
    # # find longitudinal position of conflict point
    # conflict_point = np.array([13, -2])
    # dis2cv_lt_cp = cv_lt - conflict_point
    # dis2cv_lt_cp_norm = np.linalg.norm(dis2cv_lt_cp, axis=1)
    # dis2cv_lt_cp_min = np.amin(dis2cv_lt_cp_norm)
    # cv_index_lt_cp = np.where(dis2cv_lt_cp_min == dis2cv_lt_cp_norm)
    # long_progress_lt_cp = progress_lt[cv_index_lt_cp]
    #
    # dis2cv_gs_cp = cv_gs - conflict_point
    # dis2cv_gs_cp_norm = np.linalg.norm(dis2cv_gs_cp, axis=1)
    # dis2cv_gs_cp_min = np.amin(dis2cv_gs_cp_norm)
    # cv_index_gs_cp = np.where(dis2cv_gs_cp_min == dis2cv_gs_cp_norm)
    # long_progress_gs_cp = progress_gs[cv_index_gs_cp]
    #
    # # find longitudinal progress at each time point
    # ttcp_lt = []
    # ttcp_gs = []
    # pet = []
    # for t in range(len(lt_ob_trj)):
    #     dis2cv_lt = cv_lt - lt_ob_trj[t, 0:2]
    #     dis2cv_lt_norm = np.linalg.norm(dis2cv_lt, axis=1)
    #     dis2cv_lt_min = np.amin(dis2cv_lt_norm)
    #     cv_index_lt = np.where(dis2cv_lt_min == dis2cv_lt_norm)
    #     long_progress_lt = progress_lt[cv_index_lt]
    #     ttcp_lt.append((long_progress_lt_cp - long_progress_lt) / v_lt_smoothed[t])
    #
    #     dis2cv_gs = cv_gs - gs_ob_trj[t, 0:2]
    #     dis2cv_gs_norm = np.linalg.norm(dis2cv_gs, axis=1)
    #     dis2cv_gs_min = np.amin(dis2cv_gs_norm)
    #     cv_index_gs = np.where(dis2cv_gs_min == dis2cv_gs_norm)
    #     long_progress_gs = progress_gs[cv_index_gs]
    #     ttcp_gs.append((long_progress_gs_cp - long_progress_gs) / v_gs_smoothed[t])
    #
    #     if ttcp_lt[-1] > 0 and ttcp_gs[-1] > 0:
    #         temp_pet = np.abs(ttcp_gs[-1] - ttcp_lt[-1])
    #         pet.append(float(temp_pet))
    #     else:
    #         pet.append(0)
    # pet = np.array(pet)
    "====save data to excel===="
    if save_data:
        df_lt_ob_trj = pd.DataFrame(lt_ob_trj)
        df_lt_trj_coll = pd.DataFrame(lt_trj_coll)
        df_gs_ob_trj = pd.DataFrame(gs_ob_trj)
        df_gs_trj_coll = pd.DataFrame(gs_trj_coll)
        df_link = pd.DataFrame(link)
        df_pet = pd.DataFrame(pet)
        df_estimated_gs_ipv = pd.DataFrame(agent_lt.estimated_inter_agent.ipv_collection, columns=['ipv'])
        df_estimated_gs_ipv_error = pd.DataFrame(agent_lt.estimated_inter_agent.ipv_error_collection, columns=['error'])
        df_estimated_lt_ipv = pd.DataFrame(agent_gs.estimated_inter_agent.ipv_collection, columns=['ipv'])
        df_estimated_lt_ipv_error = pd.DataFrame(agent_gs.estimated_inter_agent.ipv_error_collection, columns=['error'])

        filename_data = filedir + '/excel/output' + '_round_' + str(rd) + '_case_' + str(case_id) + '.xlsx'
        workbook = xlsxwriter.Workbook(filename_data)
        # 新增工作簿。
        worksheet = workbook.add_worksheet('lt_ob_trj')
        #  关闭工作簿。在文件夹中打开文件，查看写入的结果。
        workbook.close()  # 一定要关闭workbook后才会产生文件！

        with pd.ExcelWriter(filename_data,
                            mode='a',
                            if_sheet_exists="overlay",
                            engine="openpyxl") as writer:
            df_lt_ob_trj.to_excel(writer, index=False, sheet_name='lt_ob_trj')
            df_gs_ob_trj.to_excel(writer, index=False, sheet_name='gs_ob_trj')
            df_lt_trj_coll.to_excel(writer, index=False, sheet_name='lt_trj_coll')
            df_gs_trj_coll.to_excel(writer, index=False, sheet_name='gs_trj_coll')
            df_link.to_excel(writer, index=False, sheet_name='link')
            df_pet.to_excel(writer, index=False, sheet_name='PET')
            df_estimated_gs_ipv.to_excel(writer, index=False, startcol=0, sheet_name='ipv_gs')
            df_estimated_gs_ipv_error.to_excel(writer, index=False, startcol=1, sheet_name='ipv_gs')
            df_estimated_lt_ipv.to_excel(writer, index=False, startcol=0, sheet_name='ipv_lt')
            df_estimated_lt_ipv_error.to_excel(writer, index=False, startcol=1, sheet_name='ipv_lt')

    if save_fig or show_gif:

        # set figure
        fig = plt.figure(figsize=(12, 12))  # for showing gif
    else:
        fig = plt.figure(dpi=300, figsize=(12, 18))  # for printing figure
    ax1 = plt.subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    img = plt.imread('background_pic/T_intersection.jpg')

    "====final observed_trajectory===="
    num_frame = len(lt_ob_trj)
    for t in range(num_frame):
        ax1.cla()
        ax1.imshow(img, extent=[-9.1, 24.9, -13, 8])
        ax1.set(xlim=[-9.1, 24.9], ylim=[-13, 8])
        # central vertices
        ax1.plot(cv_lt[:, 0], cv_lt[:, 1], 'r-')
        ax1.plot(cv_gs[:, 0], cv_gs[:, 1], 'b-')
        # left-turn
        ax1.scatter(lt_ob_trj[:t + 1, 0],
                    lt_ob_trj[:t + 1, 1],
                    s=120,
                    alpha=0.4,
                    color='red',
                    label='left-turn')
        # go-straight
        ax1.scatter(gs_ob_trj[:t + 1, 0],
                    gs_ob_trj[:t + 1, 1],
                    s=120,
                    alpha=0.4,
                    color='blue',
                    label='go-straight')
        if t < len(lt_ob_trj) - 1:
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
        ax1.plot([lt_ob_trj[t, 0], gs_ob_trj[t, 0]],
                 [lt_ob_trj[t, 1], gs_ob_trj[t, 1]],
                 color='gray',
                 alpha=0.2)
        if show_gif:
            plt.pause(0.1)
    # full position link
    for t in range(len(lt_ob_trj)):
        ax1.plot([lt_ob_trj[t, 0], gs_ob_trj[t, 0]],
                 [lt_ob_trj[t, 1], gs_ob_trj[t, 1]],
                 color='gray',
                 alpha=0.1)
    ax1.set_title('gs_' + str(agent_gs.ipv) + '_lt_' + str(agent_lt.ipv), fontsize=12)

    "====ipv estimation===="
    x_range = np.array(range(len(agent_lt.estimated_inter_agent.ipv_collection)))

    # ====draw left turn
    y_lt = np.array(agent_lt.estimated_inter_agent.ipv_collection)
    point_smoothed_lt, _ = smooth_ployline(np.array([x_range, y_lt]).T)
    x_smoothed_lt = point_smoothed_lt[:, 0]
    y_lt_smoothed = point_smoothed_lt[:, 1]
    ax2.plot(x_smoothed_lt, y_lt_smoothed,
             alpha=1,
             color='blue',
             label='estimated gs IPV')
    if ipv_update_method == 1:
        # error bar
        y_error_lt = np.array(agent_lt.estimated_inter_agent.ipv_error_collection)
        error_smoothed_lt, _ = smooth_ployline(np.array([x_range, y_error_lt]).T)
        y_error_lt_smoothed = error_smoothed_lt[:, 1]
        ax2.fill_between(x_smoothed_lt, y_lt_smoothed - y_error_lt_smoothed, y_lt_smoothed + y_error_lt_smoothed,
                         alpha=0.3,
                         color='blue',
                         label='estimated gs IPV')
    # ground truth
    ax2.plot(x_range, agent_gs.ipv * np.ones_like(x_range),
             linewidth=5,
             label='actual gs IPV')

    # ====draw go straight
    y_gs = np.array(agent_gs.estimated_inter_agent.ipv_collection)
    # smoothen data
    point_smoothed_gs, _ = smooth_ployline(np.array([x_range, y_gs]).T)
    x_smoothed_gs = point_smoothed_gs[:, 0]
    y_gs_smoothed = point_smoothed_gs[:, 1]
    ax2.plot(x_smoothed_gs, y_gs_smoothed,
             alpha=1,
             color='red',
             label='estimated gs IPV')
    if ipv_update_method == 1:
        # error bar
        y_error_gs = np.array(agent_gs.estimated_inter_agent.ipv_error_collection)
        error_smoothed_gs, _ = smooth_ployline(np.array([x_range, y_error_gs]).T)
        y_error_gs_smoothed = error_smoothed_gs[:, 1]
        ax2.fill_between(x_smoothed_gs, y_gs_smoothed - y_error_gs_smoothed, y_gs_smoothed + y_error_gs_smoothed,
                         alpha=0.3,
                         color='red',
                         label='estimated lt IPV')
    # ground truth
    ax2.plot(x_range, agent_lt.ipv * np.ones_like(x_range),
             linewidth=5,
             label='actual lt IPV')

    "====PET===="
    ax3.plot(pet)

    if show_gif:
        plt.show()

    # save figure
    if save_fig:
        plt.savefig('./outputs/simulation/version' + str(version_num) + '/figures/'
                    + '_round_' + str(rd)
                    + '_case_' + str(case_id) + '.png')
        plt.clf()
        plt.close()
        gc.collect()


if __name__ == '__main__':
    # ipv_list = [-3, -2, -1, 0, 1, 2, 3]
    # ipv_list = [-2, 0, 2]
    # ipv_list = [-2]
    # for gs in [2]:
    #     for lt in ipv_list:
    #         get_results(gs, lt)
    rd = 1
    caseid = 2
    get_results(rd, caseid)
