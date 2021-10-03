import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splrep, splev
import math


def smooth_cv(cv_init, point_num=100):
    cv = cv_init
    list_x = cv[:, 0]
    list_y = cv[:, 1]
    if type(cv) is not np.ndarray:
        cv = np.array(cv)
    delta_cv = cv[1:, ] - cv[:-1, ]
    s_cv = np.linalg.norm(delta_cv, axis=1)

    s_cv = np.array([0] + list(s_cv))
    s_cv = np.cumsum(s_cv)

    bspl_x = splrep(s_cv, list_x, s=0.1)
    bspl_y = splrep(s_cv, list_y, s=0.1)
    # values for the x axis
    s_smooth = np.linspace(0, max(s_cv), point_num)
    # get y values from interpolated curve
    x_smooth = splev(s_smooth, bspl_x)
    y_smooth = splev(s_smooth, bspl_y)
    new_cv = np.array([x_smooth, y_smooth]).T

    delta_new_cv = new_cv[1:, ]-new_cv[:-1, ]
    s_accumulated = np.cumsum(np.linalg.norm(delta_new_cv, axis=1))
    s_accumulated = np.concatenate(([0], s_accumulated), axis=0)
    return new_cv, s_accumulated


def get_central_vertices(cv_type):
    cv_init = None
    if cv_type == 'lt':  # left turn
        cv_init = np.array([[0, -15], [5, -14.14], [10.6, -10.6], [15, 0], [15, 100]])
    elif cv_type == 'gs':  # go straight
        cv_init = np.array([[20, -2], [10, -2], [0, -2], [-150, -2]])
    elif cv_type == 'lt_nds':  # left turn in NDS
        cv_init = np.array([[17, 42], [20, 27], [27, 17], [40, 17]])
    elif cv_type == 'gs_nds':  # go straight in NDS
        cv_init = np.array([[18, -28], [21, 7.4], [22.6, 38.7], [24.5, 56.3]])
    assert cv_init is not None
    cv_smoothed, s_accumulated = smooth_cv(cv_init)
    return cv_smoothed, s_accumulated


def kinematic_model(u, init_state, TRACK_LEN, dt):
    if not np.size(u, 0) == TRACK_LEN - 1:
        u = np.array([u[0:TRACK_LEN - 1], u[TRACK_LEN - 1:]]).T
    r_len = 0.8
    f_len = 1
    x, y, vx, vy, h = init_state
    # track = [init_state]
    psi = h
    track = [[x, y, vx, vy, h]]
    v_temp = np.sqrt(vx ** 2 + vy ** 2)

    for i in range(len(u)):
        a = u[i][0]
        delta = u[i][1]
        beta = math.atan((r_len / (r_len + f_len)) * math.tan(delta))
        x = x + v_temp * np.cos(psi + beta) * dt
        y = y + v_temp * np.sin(psi + beta) * dt
        psi = psi + (v_temp / f_len) * np.sin(beta) * dt
        v_temp = v_temp + a * dt

        vx = v_temp*np.cos(psi)
        vy = v_temp*np.sin(psi)

        track.append([x, y, vx, vy, psi])
    return np.array(track)