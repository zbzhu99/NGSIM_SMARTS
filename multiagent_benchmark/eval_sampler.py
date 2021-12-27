import numpy as np
import multiprocessing
from collections import defaultdict


def single_env_rollout(rank, queue, env_ctor_func):
    env, eval_group_num = env_ctor_func()
    print("Process {} Started".format(rank))
    env.seed(rank)  # may not need
    paths = defaultdict(list)
    for _ in range(eval_group_num):
        obs_n = env.reset()
        vehicle_ids = env.vehicle_id
        path = {v_id: defaultdict(list) for v_id in vehicle_ids}

        done_n = {"__all__": False}
        while not done_n["__all__"]:
            # NOTE(zbzhu): here we test with random policy
            act_n = {
                a_id: np.random.uniform(0, 1, size=2)
                for a_id in obs_n.keys()
                if not done_n.get(a_id, False)
            }
            # act_n = get_action(obs_n)
            next_obs_n, rew_n, done_n, info_n = env.step(act_n)

            for a_id, info in info_n.items():
                v_id = info["vehicle_id"]
                if a_id in obs_n.keys():
                    path[v_id]["observations"].append(obs_n[a_id])
                    path[v_id]["actions"].append(act_n[a_id])
                    path[v_id]["next_observations"].append(next_obs_n[a_id])
                    path[v_id]["rewards"].append(rew_n[a_id])
                    path[v_id]["dones"].append(done_n[a_id])
                    path[v_id]["infos"].append(info_n[a_id])

            obs_n = next_obs_n

        print("{} finished".format(vehicle_ids))
        for v_id in vehicle_ids:
            if len(path[v_id]) > 0:
                paths[v_id].append(path[v_id])

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
