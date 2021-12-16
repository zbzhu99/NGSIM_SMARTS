import numpy as np
from dataclasses import replace

from example_adapter import get_observation_adapter

from smarts.core.smarts import SMARTS
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.scenario import Scenario
from smarts.core.traffic_history_provider import TrafficHistoryProvider
from smarts.env.wrappers.parallel_env import ParallelEnv


def get_action_adapter():
    def action_adapter(model_action):
        assert len(model_action) == 2
        return (model_action[0], model_action[1])

    return action_adapter


class MATrafficSim:
    def __init__(self, scenarios, agent_number, obs_stacked_size=1):
        self.scenarios_iterator = Scenario.scenario_variations(scenarios, [])
        self._init_scenario()
        self.obs_stacked_size = obs_stacked_size
        self.n_agents = agent_number
        self.agentid_to_vehid = {}
        self.agent_ids = [f"agent_{i}" for i in range(self.n_agents)]
        self.agent_spec = AgentSpec(
            interface=AgentInterface(
                max_episode_steps=None,
                waypoints=False,
                neighborhood_vehicles=True,
                ogm=False,
                rgb=False,
                lidar=False,
                action=ActionSpaceType.Imitation,
            ),
            action_adapter=get_action_adapter(),
            observation_adapter=get_observation_adapter(obs_stacked_size),
        )

        self.smarts = SMARTS(
            agent_interfaces={},
            traffic_sim=None,
            envision=None,
        )

    def seed(self, seed):
        np.random.seed(seed)

    def step(self, action):
        for agent_id in self.agent_ids:
            if agent_id not in action.keys():
                continue
            agent_action = action[agent_id]
            action[agent_id] = self.agent_spec.action_adapter(agent_action)
        observations, rewards, dones, _ = self.smarts.step(action)
        info = {}

        for k in observations.keys():
            observations[k] = self.agent_spec.observation_adapter(
                observations[k]
            )

        dones["__all__"] = all(dones.values())

        return (
            observations,
            rewards,
            dones,
            info,
        )

    def reset(self):
        if self.vehicle_itr + self.n_agents >= (len(self.vehicle_ids) - 1):
            self.vehicle_itr = 0

        self.vehicle_id = self.vehicle_ids[
            self.vehicle_itr : self.vehicle_itr + self.n_agents
        ]

        traffic_history_provider = self.smarts.get_provider_by_type(
            TrafficHistoryProvider
        )
        assert traffic_history_provider

        for i in range(self.n_agents):
            self.agentid_to_vehid[f"agent_{i}"] = self.vehicle_id[i]

        agent_interfaces = {}
        history_start_time = self.vehicle_missions[self.vehicle_id[0]].start_time
        for agent_id in self.agent_ids:
            vehicle = self.agentid_to_vehid[agent_id]
            agent_interfaces[agent_id] = self.agent_spec.interface
            if history_start_time > self.vehicle_missions[vehicle].start_time:
                history_start_time = self.vehicle_missions[vehicle].start_time

        traffic_history_provider.start_time = history_start_time
        ego_missions = {}
        for agent_id in self.agent_ids:
            vehicle = self.agentid_to_vehid[agent_id]
            ego_missions[agent_id] = replace(
                self.vehicle_missions[vehicle],
                start_time=self.vehicle_missions[vehicle].start_time
                - history_start_time,
            )
        self.scenario.set_ego_missions(ego_missions)
        self.smarts.switch_ego_agents(agent_interfaces)

        observations = self.smarts.reset(self.scenario)
        for k in observations.keys():
            observations[k] = self.agent_spec.observation_adapter(
                observations[k]
            )

        return observations

    def _init_scenario(self):
        self.scenario = next(self.scenarios_iterator)
        self.vehicle_missions = self.scenario.discover_missions_of_traffic_histories()
        self.veh_start_times = {}
        for v_id, mission in self.vehicle_missions.items():
            self.veh_start_times[v_id] = mission.start_time
        self.vehicle_ids = list(self.vehicle_missions.keys())
        vlist = []
        for vehicle_id, start_time in self.veh_start_times.items():
            vlist.append((vehicle_id, start_time))
        dtype = [("id", int), ("start_time", float)]
        vlist = np.array(vlist, dtype=dtype)
        vlist = np.sort(vlist, order="start_time")
        self.vehicle_ids = list(self.vehicle_missions.keys())
        for id in range(len(self.vehicle_ids)):
            self.vehicle_ids[id] = f"{vlist[id][0]}"
        self.vehicle_itr = 0

    def close(self):
        if self.smarts is not None:
            self.smarts.destroy()


if __name__ == "__main__":
    """ Dummy Rollout """
    env = MATrafficSim(["./ngsim"], agent_number=5)
    obs = env.reset()
    done = {}
    n_steps = 10
    for step in range(n_steps):
        act_n = {}
        for agent_id in obs.keys():
            if step and done[agent_id]:
                continue
            act_n[agent_id] = np.random.normal(0, 1, size=(2,))
        obs, rew, done, info = env.step(act_n)
        print(rew)
    print("finished")
    env.close()

    """ Parallel Rollout """
    env_num = 2
    env_creator = lambda: MATrafficSim(["./ngsim"], agent_number=5)
    vector_env = ParallelEnv([env_creator] * env_num, auto_reset=True)

    vec_obs = vector_env.reset()

    vec_act = []
    for obs in vec_obs:
        vec_act.append({a_id: np.random.normal(0, 1, size=(2,)) for a_id in obs.keys()})

    vec_next_obs, vec_rew, vec_done, vec_info = vector_env.step(vec_act)

    print("parallel finished!")
    vector_env.close()
