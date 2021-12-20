import numpy as np
from dataclasses import replace
from example_adapter import get_observation_adapter

from smarts.core.smarts import SMARTS
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.scenario import Scenario
from smarts.core.traffic_history_provider import TrafficHistoryProvider


def get_action_adapter():
    def action_adapter(model_action):
        assert len(model_action) == 2
        return (model_action[0], model_action[1])

    return action_adapter


class TrafficSim:
    def __init__(self, scenarios, obs_stacked_size=1):
        self.scenarios_iterator = Scenario.scenario_variations(scenarios, [])
        self._init_scenario()
        self.obs_stacked_size = obs_stacked_size
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

        raw_observations, rewards, dones, _ = self.smarts.step(
            {self.vehicle_id: self.agent_spec.action_adapter(action)}
        )

        observation = self.agent_spec.observation_adapter(
            raw_observations[self.vehicle_id]
        )

        return (
            observation,
            rewards[self.vehicle_id],
            {"__all__": dones[self.vehicle_id]},
            {
                "vehicle_id": self.vehicle_id,
                "reached_goal": raw_observations[self.vehicle_id].events.reached_goal,
            },
        )

    def reset(self):
        if self.vehicle_itr >= len(self.vehicle_ids):
            self.vehicle_itr = 0

        self.vehicle_id = self.vehicle_ids[self.vehicle_itr]
        vehicle_mission = self.vehicle_missions[self.vehicle_id]

        traffic_history_provider = self.smarts.get_provider_by_type(
            TrafficHistoryProvider
        )
        assert traffic_history_provider
        traffic_history_provider.start_time = vehicle_mission.start_time

        modified_mission = replace(vehicle_mission, start_time=0.0)
        self.scenario.set_ego_missions({self.vehicle_id: modified_mission})
        self.smarts.switch_ego_agents({self.vehicle_id: self.agent_spec.interface})

        raw_observations = self.smarts.reset(self.scenario)
        observation = self.agent_spec.observation_adapter(
            raw_observations[self.vehicle_id]
        )
        self.vehicle_itr += 1
        return observation

    def _init_scenario(self):
        self.scenario = next(self.scenarios_iterator)
        self.vehicle_missions = self.scenario.discover_missions_of_traffic_histories()
        self.vehicle_ids = list(self.vehicle_missions.keys())
        np.random.shuffle(self.vehicle_ids)
        self.vehicle_itr = 0

    def close(self):
        if self.smarts is not None:
            self.smarts.destroy()
