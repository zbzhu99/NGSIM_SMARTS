import numpy as np

from traffic_simulator import TrafficSim


if __name__ == "__main__":
    env = TrafficSim(["./ngsim"])
    obs = env.reset()

    act = np.random.normal(0, 1, size=(2,))
    obs, rew, done, info = env.step(act)

    print("finished")
    env.close()
