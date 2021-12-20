import os
import sys
import argparse
import pickle
import sqlite3
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR.parent))

from eval_sampler import ParallelPathSampler
from fixed_traffic_simulator import FixedTrafficSim
from math_utils import DiscreteFrechet

sys.setrecursionlimit(25000)


def benchmark(vehicle_ids_list, ngsim_path):
    sampler = ParallelPathSampler(
        [make_env(vehicle_id, ngsim_path) for vehicle_id in vehicle_ids_list]
    )
    rollout_trajs = sampler.collect_samples()
    compute_metrics(rollout_trajs, ngsim_path)


def compute_metrics(rollout_trajs, ngsim_path):
    def _compute_frechet_distance(sample_traj, expert_traj):
        frechet_solver = DiscreteFrechet(
            dist_func=lambda p, q: np.linalg.norm(p - q)
        )
        frechet_distance = frechet_solver.distance(
            np.stack([traj[:2] for traj in expert_traj], axis=0),
            np.concatenate([traj["ego_pos"] for traj in sample_traj["observations"]], axis=0),
        )
        return frechet_distance

    def _compute_distance_travelled(sample_traj):
        # NOTE(zbzhu): we only care about the distance in horizontal direction
        distance_travelled = abs(
            sample_traj["observations"][-1]["ego_pos"][0]
            - sample_traj["observations"][0]["ego_pos"][0]
        )
        return distance_travelled

    def _judge_success(sample_traj):
        if sample_traj["infos"][-1]["reached_goal"]:
            return 1.0
        else:
            return 0.0

    metrics = defaultdict(list)
    demo_path = Path(ngsim_path) / "i80_0400-0415.shf"
    dbconxn = sqlite3.connect(demo_path)
    cur = dbconxn.cursor()
    for vehicle_id in rollout_trajs.keys():
        query = """SELECT position_x, position_y, heading_rad, speed
                    FROM Trajectory
                    WHERE vehicle_id = ?
                    """
        cur.execute(query, [vehicle_id])
        expert_traj = cur.fetchall()
        sample_traj = rollout_trajs[vehicle_id]
        metrics["Frechet Distance"].append(_compute_frechet_distance(sample_traj, expert_traj))
        metrics["Distance Travelled"].append(_compute_distance_travelled(sample_traj))
        metrics["Success"].append(_judge_success(sample_traj))
    cur.close()

    print("Average Frechet Distance: {}".format(np.mean(metrics["Frechet Distance"])))
    print("Average Distance Travelled: {}".format(np.mean(metrics["Distance Travelled"])))
    print("Success Rate: {}".format(np.mean(metrics["Success"])))


def make_env(vehicle_ids, ngsim_path):
    def _init():
        return FixedTrafficSim(scenarios=[ngsim_path], vehicle_ids=vehicle_ids), len(vehicle_ids)
    return _init


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ngsim_path",
        type=str,
        help="Path to your ngsim scenario folder"
    )
    parser.add_argument(
        "--env_num",
        default=30,
        type=int,
        help="Num of parallel environments for sampling"
    )
    args = parser.parse_args()

    with open(os.path.join(BASE_DIR, "test_ids.pkl"), "rb") as f:
        eval_vehicle_ids = pickle.load(f)
    vehicle_ids_list = np.array_split(
        eval_vehicle_ids,
        args.env_num,
    )

    benchmark(vehicle_ids_list, args.ngsim_path)
