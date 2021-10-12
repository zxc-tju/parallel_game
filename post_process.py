import pickle
import math
from matplotlib import pyplot as plt
from tools.utility import get_central_vertices, smooth_cv
import numpy as np

"====import data===="
gs_ipv = 3
lt_ipv = 1

ipv_update_method = 1
filename = './outputs/version11/' + 'agents_info' \
           + '_gs_' + str(gs_ipv) \
           + '_lt_' + str(lt_ipv) \
           + '.pckl'

f = open(filename, 'rb')
agent_lt, agent_gs = pickle.load(f)
f.close()

cv_init_it, _ = get_central_vertices('lt')
cv_init_gs, _ = get_central_vertices('gs')

if hasattr(agent_lt, 'trajectory'):
    agent_lt_observed_trajectory = agent_lt.trajectory
    agent_gs_observed_trajectory = agent_gs.trajectory
else:
    agent_lt_observed_trajectory = agent_lt.observed_trajectory
    agent_gs_observed_trajectory = agent_gs.observed_trajectory

"====final observed_trajectory===="
fig = plt.figure(1)
ax = fig.add_subplot(111)
img = plt.imread('background_pic/T_intersection.jpg')

for t in range(len(agent_lt_observed_trajectory)):
    ax.cla()
    ax.imshow(img, extent=[-9.1, 24.9, -13, 8])
    # central vertices
    ax.plot(cv_init_it[:, 0], cv_init_it[:, 1], 'r-')
    ax.plot(cv_init_gs[:, 0], cv_init_gs[:, 1], 'b-')
    # left-turn
    ax.scatter(agent_lt_observed_trajectory[:t + 1, 0],
               agent_lt_observed_trajectory[:t + 1, 1],
               s=100,
               alpha=0.6,
               color='red',
               label='left-turn')

    # go-straight
    ax.scatter(agent_gs_observed_trajectory[:t + 1, 0],
               agent_gs_observed_trajectory[:t + 1, 1],
               s=100,
               alpha=0.6,
               color='blue',
               label='go-straight')

    if t < len(agent_lt_observed_trajectory) - 1:
        # real-time virtual plans of ## ego ## at time step t
        lt_track = agent_lt.trj_solution_collection[t]
        ax.plot(lt_track[:, 0], lt_track[:, 1], '--', linewidth=3)
        gs_track = agent_gs.trj_solution_collection[t]
        ax.plot(gs_track[:, 0], gs_track[:, 1], '--', linewidth=3)

        if ipv_update_method == 1:
            # real-time virtual plans of ## interacting agent ## at time step t
            candidates_lt = agent_lt.estimated_inter_agent.virtual_track_collection[t]
            candidates_gs = agent_gs.estimated_inter_agent.virtual_track_collection[t]
            for track_lt in candidates_lt:
                ax.plot(track_lt[:, 0], track_lt[:, 1], color='green', alpha=0.5)
            for track_gs in candidates_gs:
                ax.plot(track_gs[:, 0], track_gs[:, 1], color='green', alpha=0.5)

    # position link
    ax.plot([agent_lt_observed_trajectory[t, 0], agent_gs_observed_trajectory[t, 0]],
            [agent_lt_observed_trajectory[t, 1], agent_gs_observed_trajectory[t, 1]],
            color='gray',
            alpha=0.1)
    ax.set(xlim=[-9.1, 24.9], ylim=[-13, 8])
    plt.pause(0.3)

for t in range(len(agent_lt_observed_trajectory)):
    ax.plot([agent_lt_observed_trajectory[t, 0], agent_gs_observed_trajectory[t, 0]],
            [agent_lt_observed_trajectory[t, 1], agent_gs_observed_trajectory[t, 1]],
            color='gray',
            alpha=0.1)

"====ipv estimation===="
plt.figure(2)
x_range = np.array(range(len(agent_lt.estimated_inter_agent.ipv_collection)))

# ====draw left turn
y_lt = np.array(agent_lt.estimated_inter_agent.ipv_collection)
point_smoothed_lt, _ = smooth_cv(np.array([x_range, y_lt]).T)
x_smoothed_lt = point_smoothed_lt[:, 0]
y_lt_smoothed = point_smoothed_lt[:, 1]
plt.plot(x_smoothed_lt, y_lt_smoothed,
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
plt.plot(x_range, gs_ipv * math.pi / 8 * np.ones_like(x_range),
         linewidth=5,
         label='actual gs IPV')

# ====draw go straight
y_gs = np.array(agent_gs.estimated_inter_agent.ipv_collection)
# smoothen data
point_smoothed_gs, _ = smooth_cv(np.array([x_range, y_gs]).T)
x_smoothed_gs = point_smoothed_gs[:, 0]
y_gs_smoothed = point_smoothed_gs[:, 1]
plt.plot(x_smoothed_gs, y_gs_smoothed,
         alpha=1,
         color='red',
         label='estimated gs IPV')

if ipv_update_method == 1:
    # error bar
    y_error_gs = np.array(agent_gs.estimated_inter_agent.ipv_error_collection)
    error_smoothed_gs, _ = smooth_cv(np.array([x_range, y_error_gs]).T)
    y_error_gs_smoothed = error_smoothed_gs[:, 1]
    plt.fill_between(x_smoothed_gs, y_gs_smoothed - y_error_gs_smoothed, y_gs_smoothed + y_error_gs_smoothed,
                     alpha=0.3,
                     color='red',
                     label='estimated lt IPV')
# ground truth
plt.plot(x_range, lt_ipv * math.pi / 8 * np.ones_like(x_range),
         linewidth=5,
         label='actual lt IPV')
plt.show()
