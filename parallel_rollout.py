import sys
import numpy as np

from traffic_simulator import TrafficSim
from smarts.env.wrappers.parallel_env import ParallelEnv

# Increase system recursion limit
sys.setrecursionlimit(25000)


if __name__ == "__main__":
    env_num = 2
    env_creator = lambda: TrafficSim(["./ngsim"])
    vector_env = ParallelEnv([env_creator] * env_num, auto_reset=True)

    vec_obs = vector_env.reset()

    vec_act = [np.random.normal(0, 1, size=(2,)) for _ in range(env_num)]

    vec_next_obs, vec_rew, vec_done, vec_info = vector_env.step(vec_act)

    print("finished!")
    vector_env.close()
