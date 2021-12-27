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
from fixed_traffic_simulator import FixedMATrafficSim
from math_utils import DiscreteFrechet

sys.setrecursionlimit(25000)


def benchmark(vehicle_id_groups_list, agent_num, ngsim_path):
    sampler = ParallelPathSampler(
        [make_env(vehicle_id_groups, agent_num, ngsim_path) for vehicle_id_groups in vehicle_id_groups_list]
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
            sample_traj["observations"][-1]["ego_pos"][0][0]
            - sample_traj["observations"][0]["ego_pos"][0][0]
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
        for sample_traj in rollout_trajs[vehicle_id]:
            metrics["Frechet Distance"].append(_compute_frechet_distance(sample_traj, expert_traj))
            metrics["Distance Travelled"].append(_compute_distance_travelled(sample_traj))
            metrics["Success"].append(_judge_success(sample_traj))
    cur.close()

    print("Average Frechet Distance: {}".format(np.mean(metrics["Frechet Distance"])))
    print("Average Distance Travelled: {}".format(np.mean(metrics["Distance Travelled"])))
    print("Success Rate: {}".format(np.mean(metrics["Success"])))


def make_env(vehicle_id_groups, agent_number, ngsim_path):
    def _init():
        return FixedMATrafficSim(
            scenarios=[ngsim_path],
            agent_number=agent_number,
            vehicle_id_groups=vehicle_id_groups,
        ), len(vehicle_id_groups)
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
        default=10,
        type=int,
        help="Num of parallel environments for sampling"
    )
    parser.add_argument(
        "--agent_num",
        default=5,
        type=int,
        help="Num of vehicles controlled during sampling"
    )
    args = parser.parse_args()

    with open(os.path.join(BASE_DIR, "test_ids.pkl"), "rb") as f:
        eval_vehicle_ids = pickle.load(f)

    eval_vehicle_groups = []
    for idx in range(len(eval_vehicle_ids) - args.agent_num):
        eval_vehicle_groups.append(eval_vehicle_ids[idx: idx + args.agent_num])

    vehicle_id_groups_list = np.array_split(
        eval_vehicle_groups,
        args.env_num,
    )

    benchmark(vehicle_id_groups_list, args.agent_num, args.ngsim_path)
