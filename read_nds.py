"""
for visualize driving trajectories in the Jianhe-Xianxia Intersection
"""
import scipy.io
import numpy as np
from matplotlib import pyplot as plt


def vis_nds(case_id='all'):
    if case_id == 'all':
        target_range = range(130)
    else:
        target_range = {case_id}

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(xlim=[-22, 53], ylim=[-31, 57])
    img = plt.imread('background_pic/Jianhexianxia.jpg')
    ax.imshow(img, extent=[-22, 53, -31, 57])

    for i in target_range:
        # abstract interaction info. of a given case
        case_info = inter_info[i]
        # left-turn vehicle
        lt_info = case_info[0]
        # go-straight vehicles
        gs_info_multi = case_info[1:inter_num[0, i] + 1]

        # go straight trajectories
        for gs_id in range(np.size(gs_info_multi, 0)):

            # delete invalid (0,0) positions
            gs_trj_temp = gs_info_multi[gs_id][:, 0:2]
            invalid_len = len((np.where(gs_trj_temp[:, 0] == 0))[0])

            plt.plot(gs_info_multi[gs_id][invalid_len:, 0], gs_info_multi[gs_id][invalid_len:, 1],
                     alpha=0.5,
                     color='red')

        # left-turn trajectories
        ax.plot(lt_info[:, 0], lt_info[:, 1],
                alpha=0.5,
                color='blue')
    plt.show()


if __name__ == '__main__':
    # load mat file
    mat = scipy.io.loadmat('./data/NDS_data.mat')

    # full interaction information
    inter_info = mat['interaction_info']
    '''
    inter_info: (131 scenarios) x (less than 24 vehicles)
    in each scenario, the first vehicle was turning left and others were going straight
    for each driver, info. in column are as follow:
    0-1: [position x] [position y]
    2-3: [velocity x] [velocity y]
    4: [heading]
    5: [velocity overall = sqrt(vx^2+xy^2)]
    6: [curvature] (only for left-turn vehicles)
    dt = 0.12s 
    '''

    # the number of go-straight vehicles that interact with the left-turn vehicle in each scenario
    inter_num = mat['interact_agent_num']

    vis_nds()
