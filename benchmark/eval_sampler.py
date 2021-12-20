import numpy as np
import multiprocessing
from collections import defaultdict


def single_env_rollout(rank, queue, env_ctor_func):
    env, eval_car_num = env_ctor_func()
    print("Process {} Started".format(rank))
    env.seed(rank)  # may not need
    paths = {}
    for _ in range(eval_car_num):
        path = defaultdict(list)
        obs = env.reset()
        done = False
        while not done:
            # NOTE(zbzhu): here we test with random policy
            act = np.random.uniform(0, 1, size=2)
            # act = get_action(obs)
            next_obs, rew, done, info = env.step(act)
            done = done["__all__"]

            path["observations"].append(obs)
            path["actions"].append(act)
            path["next_observations"].append(next_obs)
            path["rewards"].append(rew)
            path["dones"].append(done)
            path["infos"].append(info)

            obs = next_obs

        vehicle_id = path["infos"][-1]["vehicle_id"]
        print("{} finished".format(vehicle_id))
        paths[vehicle_id] = path

    queue.put([rank, paths])
    print("Process {} Ended".format(rank))


class ParallelPathSampler:
    def __init__(
        self,
        env_ctor_func_list,
    ):
        self.env_ctor_func_list = env_ctor_func_list

    def collect_samples(self):
        worker_num = len(self.env_ctor_func_list)
        queue = multiprocessing.Queue()
        workers = []
        for i in range(worker_num):
            worker_args = (i, queue, self.env_ctor_func_list[i])
            workers.append(multiprocessing.Process(target=single_env_rollout, args=worker_args))

        for worker in workers:
            worker.start()

        paths = {}
        for _ in workers:
            pid, _paths = queue.get()
            paths = {**paths, **_paths}

        return paths
