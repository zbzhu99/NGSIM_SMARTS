import copy
import numpy as np
import pickle
import argparse

from example_adapter import get_observation_adapter

from smarts.core.smarts import SMARTS
from smarts.core.scenario import Scenario
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType


def acceleration_count(obs, obs_next, acc_dict, ang_v_dict, avg_dis_dict):
    acc_dict = {}
    for car in obs.keys():
        car_state = obs[car].ego_vehicle_state
        angular_velocity = car_state.yaw_rate
        ang_v_dict.append(angular_velocity)
        dis_cal = car_state.speed * 0.1
        if car in avg_dis_dict:
            avg_dis_dict[car] += dis_cal
        else:
            avg_dis_dict[car] = dis_cal
        if car not in obs_next.keys():
            continue
        car_next_state = obs_next[car].ego_vehicle_state
        acc_cal = (car_next_state.speed - car_state.speed) / 0.1
        acc_dict.append(acc_cal)


def cal_action(obs, obs_next, dt=0.1):
    act = {}
    for car in obs.keys():
        if car not in obs_next.keys():
            continue
        car_state = obs[car].ego_vehicle_state
        car_next_state = obs_next[car].ego_vehicle_state
        acceleration = (car_next_state.speed - car_state.speed) / dt
        angular_velocity = car_state.yaw_rate
        act[car] = np.array([acceleration, angular_velocity])
    return act


def main(scenario, obs_stack_size=1):
    """Collect expert observations.

    Each input scenario is associated with some trajectory files. These trajectories
    will be replayed on SMARTS and observations of each vehicle will be collected and
    stored in a dict.

    Args:
        scenarios: A string of the path to scenarios to be processed.

    Returns:
        A dict in the form of {"observation": [...], "next_observation": [...], "done": [...]}.
    """

    agent_spec = AgentSpec(
        interface=AgentInterface(
            max_episode_steps=None,
            waypoints=False,
            neighborhood_vehicles=True,
            ogm=False,
            rgb=False,
            lidar=False,
            action=ActionSpaceType.Imitation,
        ),
        observation_adapter=get_observation_adapter(obs_stack_size),
    )

    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=None,
    )
    scenarios_iterator = Scenario.scenario_variations(
        [scenario],
        list([]),
    )

    smarts.reset(next(scenarios_iterator))

    expert_obs = []
    expert_acts = []
    expert_obs_next = []
    expert_terminals = []
    cars_obs = {}
    cars_act = {}
    cars_obs_next = {}
    cars_terminals = {}

    prev_vehicles = set()
    done_vehicles = set()
    prev_obs = None
    while True:
        smarts.step({})

        current_vehicles = smarts.vehicle_index.social_vehicle_ids()
        done_vehicles = prev_vehicles - current_vehicles
        prev_vehicles = current_vehicles

        if len(current_vehicles) == 0:
            break

        smarts.attach_sensors_to_vehicles(
            agent_spec, smarts.vehicle_index.social_vehicle_ids()
        )
        obs, _, _, dones = smarts.observe_from(
            smarts.vehicle_index.social_vehicle_ids()
        )

        for v in done_vehicles:
            cars_terminals[f"Agent-{v}"][-1] = True
            print(f"Agent-{v} Ended")

        # handle actions
        if prev_obs is not None:
            act = cal_action(prev_obs, obs)
            for car in act.keys():
                if cars_act.__contains__(car):
                    cars_act[car].append(act[car])
                else:
                    cars_act[car] = [act[car]]
        prev_obs = copy.copy(obs)

        # handle observations
        cars = obs.keys()
        for car in cars:
            _obs = agent_spec.observation_adapter(obs[car])
            if cars_obs.__contains__(car):
                cars_obs[car].append(_obs)
                cars_terminals[car].append(dones[car])
            else:
                cars_obs[car] = [_obs]
                cars_terminals[car] = [dones[car]]

    for car in cars_obs:
        cars_obs_next[car] = cars_obs[car][1:]
        cars_obs[car] = cars_obs[car][:-1]
        cars_act[car] = np.array(cars_act[car])
        cars_terminals[car] = np.array(cars_terminals[car][:-1])
        expert_obs.append(cars_obs[car])
        expert_acts.append(cars_act[car])
        expert_obs_next.append(cars_obs_next[car])
        expert_terminals.append(cars_terminals[car])

    with open("expert.pkl", "wb") as f:
        pickle.dump(
            {
                "observations": expert_obs,
                "actions": expert_acts,
                "next_observations": expert_obs_next,
                "terminals": expert_terminals,
            },
            f,
        )

    smarts.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        default="./ngsim",
    )
    args = parser.parse_args()
    main(scenario=args.scenario)
