import math
from Agent import Agent
import numpy as np

INITIAL_IPV = math.pi / 4


def simulate():
    # initial state of the left-turn vehicle
    init_position_lt = np.array([3, -9])
    init_velocity_lt = np.array([2, 0.3])
    init_heading_lt = np.array([0])
    # initial state of the go-straight vehicle
    init_position_gs = np.array([18, -2])
    init_velocity_gs = np.array([-2, 0])
    init_heading_gs = np.array([math.pi])

    # generate LT and GS agents
    agent_lt = Agent(init_position_lt, init_velocity_lt, init_heading_lt, 'lt')
    agent_lt.ipv = INITIAL_IPV
    agent_gs = Agent(init_position_gs, init_velocity_gs, init_heading_gs, 'gs')
    agent_gs.ipv = INITIAL_IPV

    while True:
        agent_lt.solve_game(agent_gs)
        agent_gs.solve_game(agent_lt)


if __name__ == '__main__':
    simulate()
