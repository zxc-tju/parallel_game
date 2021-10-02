import scipy.io
import math
import numpy as np
from agent import Agent

mat = scipy.io.loadmat('./data/NDS_data.mat')
inter_num = mat['interactagentnumpost']
inter_info = mat['interactinfo']


# class Scenario:
#     def __init__(self):
#         self.lt_agent = Agent()


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
    for i in range(np.size(gs_info_multi, 0)):
        if inter_d[i] - inter_o[i] > 0:  # interaction is solid
            # generate two agents
            init_position_gs = gs_info_multi[i][int(inter_o[i]), 0:2]
            init_velocity_gs = gs_info_multi[i][int(inter_o[i]), 2:4]
            init_heading_gs = gs_info_multi[i][int(inter_o[i]), 4]
            agent_gs = Agent(init_position_gs, init_velocity_gs, init_heading_gs, 'gs_nds')
            agent_gs.observed_trajectory = gs_info_multi[i][int(inter_o[i]):int(inter_d[i]), 0:2]

            init_position_lt = lt_info[int(inter_o[i]), 0:2]
            init_velocity_lt = lt_info[int(inter_o[i]), 2:4]
            init_heading_lt = lt_info[int(inter_o[i]), 4]
            agent_lt = Agent(init_position_lt, init_velocity_lt, init_heading_lt, 'lt_nds')
            agent_gs.observed_trajectory = lt_info[int(inter_o[i]):int(inter_d[i]), 0:2]

            if inter_d[i] - inter_o[i] <= 6:
                continue
            else:
                for t in range(int(inter_o[i]), int(inter_d[i])-6):
                    # generate solution for each other, given self solution is published
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


if __name__ == '__main__':
    analyze_nds(1)
