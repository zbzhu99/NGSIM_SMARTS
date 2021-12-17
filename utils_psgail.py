import sys
import numpy as np
import torch
from multiagent_traffic_simulator import MATrafficSim
from smarts.env.wrappers.parallel_env import ParallelEnv
import random
import matplotlib.pyplot as plt

# Increase system recursion limit
sys.setrecursionlimit(25000)
device1 = "cuda:0"
device0 = "cpu"

def getlist(list_, idx):
    if idx < 0 or idx >= len(list_) or len(list_) == 0:
        return None
    else:
        return list_[idx]

def smooth_curve(y, smooth):
    r = smooth
    length = int(np.prod(y.shape))
    for i in range(length):
        if i > 0:
            if (not np.isinf(y[i - 1])) and (not np.isnan(y[i - 1])):
                y[i] = y[i - 1] * r + y[i] * (1 - r)
    return y

def moving_average(y, x=None, total_steps=100, smooth=0.9, move_max=False):
    if isinstance(y, list):
        y = np.array(y)
    length = int(np.prod(y.shape))
    if x is None:
        x = list(range(1, length+1))
    if isinstance(x, list):
        x = np.array(x)
    if length > total_steps:
        block_size = length//total_steps
        select_list = list(range(0, length, block_size))
        select_list = select_list[:-1]
        y = y[:len(select_list) * block_size].reshape(-1, block_size)
        if move_max:
            y = np.max(y, -1)
        else:
            y = np.mean(y, -1)
        x = x[select_list]
    y = smooth_curve(y, smooth)
    return y, x
def plotReward(infos):
    x, y = infos["episodes"],infos["rewards"]
    y, x = moving_average(y, x)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(x, y)
    plt.show()


class trajectory():
    def __init__(self):
        self.states = []
        self.actions = []
        self.hiddens = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.probs = []


class samples_agents():
    def __init__(self):
        self.states = []
        self.actions = []
        self.hiddens = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.probs = []


def dump_trajectory(expert_trajectory, agent_id, batch_samples):
    batch_samples.states += expert_trajectory[agent_id].states
    batch_samples.probs += expert_trajectory[agent_id].probs
    batch_samples.actions += expert_trajectory[agent_id].actions
    batch_samples.hiddens += expert_trajectory[agent_id].hiddens
    batch_samples.next_states += expert_trajectory[agent_id].next_states
    batch_samples.rewards += expert_trajectory[agent_id].rewards
    batch_samples.dones += expert_trajectory[agent_id].dones


def dump_all(expert_trajectory, batch_samples):
    for expert_trajectory in expert_trajectory.values():
        for agent_id in expert_trajectory.keys():
            batch_samples.states += expert_trajectory[agent_id].states
            batch_samples.probs += expert_trajectory[agent_id].probs
            batch_samples.actions += expert_trajectory[agent_id].actions
            batch_samples.hiddens += expert_trajectory[agent_id].hiddens
            batch_samples.next_states += expert_trajectory[agent_id].next_states
            batch_samples.rewards += expert_trajectory[agent_id].rewards
            batch_samples.dones += expert_trajectory[agent_id].dones


def trans2tensor(batch):
    for k in batch:
        if k == 'action' or k == 'probs':
            batch[k] = torch.cat(batch[k], dim=0).to(device1)
        else:
            batch[k] = torch.tensor(batch[k], device=device1, dtype=torch.float32)
    return batch


def obs_extractor(obs_from_agent):
    if obs_from_agent is None:
        return np.zeros(36)
    obs_vector = np.concatenate((obs_from_agent['ego_pos'],
                                 obs_from_agent['heading'],
                                 obs_from_agent['speed'],
                                 obs_from_agent['neighbor']), axis=1)
    return obs_vector


def sampling(psgail, vector_env, batch_size):
    vector_env.seed(random.randint(1, 500))
    vec_obs = vector_env.reset()
    vec_done = []
    states = []
    actions = []
    rewards = []
    next_states = []
    probs = []
    dones = []
    counter = 0
    while True:
        vec_act = []
        obs_vectors = np.zeros((1, 36))
        for idx, obs in enumerate(vec_obs):
            for agent_id in obs.keys():
                if getlist(vec_done, idx) is not None and vec_done[idx].get(agent_id):
                    continue
                obs_vectors = np.vstack((obs_vectors, obs_extractor(obs[agent_id])))
                states.append(obs_vectors[-1, :])
        obs_vectors = torch.tensor([obs_vectors[1:]], device=device1, dtype=torch.float32)
        log_prob, prob, acts = psgail.get_action(obs_vectors.squeeze())
        act_idx = 0
        prob = prob.to(device0)
        acts = acts.to(device0)
        for idx, obs in enumerate(vec_obs):
            act_n = {}
            for agent_id in obs.keys():
                if getlist(vec_done, idx) is not None and vec_done[idx].get(agent_id):
                    continue
                act_tmp = acts[act_idx].cpu()
                act_n[agent_id] = act_tmp.numpy()
                act_idx += 1
            vec_act.append(act_n)
        probs.append(prob)
        actions.append(acts)
        vec_obs, vec_rew, vec_done, vec_info = vector_env.step(vec_act)
        for idx, act_n in enumerate(vec_act):
            for agent_id in act_n.keys():
                obs_vectors = obs_extractor(vec_obs[idx].get(agent_id))
                next_states.append(obs_vectors.squeeze())
                rewards.append(vec_rew[idx].get(agent_id))
                dones.append(vec_done[idx].get(agent_id))
                counter += 1
        if counter >= batch_size:
            break
    return states, next_states, actions, probs, dones, rewards


# def sampling(psgail, sap_size=10000, env_num=12, agent_number=10):
#     env_creator = lambda: MATrafficSim(["./ngsim"], agent_number=agent_number)
#     vector_env = ParallelEnv([env_creator] * env_num, auto_reset=True)
#     vec_obs = vector_env.reset()
#     vec_done = []
#     states = []
#     acts = []
#     rewards = []
#     next_states = []
#     probs = []
#     dones = []
#     while True:
#         vec_act = []
#         for idx, obs in enumerate(vec_obs):
#             act_n = {}
#             obs_vectors = {}
#             for agent_id in obs.keys():
#                 if (getlist(vec_done, idx) is not None and vec_done[idx][agent_id]):
#                     continue
#                 obs_vectors[agent_id] = obs_extractor(obs[agent_id])
#                 states.append(obs_vectors)
#                 log_prob, prob, act_n[agent_id] = psgail.get_action(obs_vectors)
#                 acts.append(act_n[agent_id])
#                 probs.append(prob)
#             vec_act.append(act_n)
#         vec_obs, vec_rew, vec_done, vec_info = vector_env.step(vec_act)
#         for idx, obs in enumerate(vec_obs):
#             for agent_id in vec_act[idx].keys():
#                 obs_vectors = obs_extractor(vec_obs[idx].get(agent_id))
#                 next_states.append(obs_vectors)
#                 rewards.append(vec_rew[idx].get(agent_id))
#                 dones.append(vec_done[idx].get(agent_id))
#         if len(dones) >= sap_size:
#             break
#     vector_env.close()
#     return states, next_states, acts, probs, dones, rewards

# def assign_neighbors(neighbors, targets, relative_pos, idx):
#     if abs(relative_pos[0]) < abs(targets[1]):
#         targets[1] = relative_pos[0]
#         neighbors[1] = idx
#     elif targets[0] < relative_pos[0] < targets[1]:
#         targets[0] = relative_pos[0]
#         neighbors[0] = idx
#     elif targets[1] < relative_pos[0] < targets[2]:
#         targets[2] = relative_pos[0]
#         neighbors[2] = idx
#
#
# def obs_extractor(obs):
#     if obs is None:
#         return None
#     ego_vehicle_state = obs.ego_vehicle_state
#     neighborhood_vehicle_states = obs.neighborhood_vehicle_states
#     neighbors_up_idx = -np.ones(3).astype(int)
#     neighbors_middle_idx = -np.ones(3).astype(int)
#     neighbors_down_idx = -np.ones(3).astype(int)
#     neighbors_up = np.zeros((3, 4)).astype(float)
#     neighbors_middle = np.zeros((3, 4)).astype(float)
#     neighbors_down = np.zeros((3, 4)).astype(float)
#     center_lane = ego_vehicle_state.lane_index
#     targets_up = np.array([-10000, -10000, 10000])
#     targets_middle = np.array([-10000, 0, 10000])
#     targets_down = np.array([-10000, -10000, 10000])
#     for idx, info in enumerate(neighborhood_vehicle_states):
#         relative_pos = info[1][:-1] - ego_vehicle_state[1][:-1]
#         if info.lane_index == center_lane + 1:
#             assign_neighbors(neighbors_up_idx, targets_up, relative_pos, idx)
#         elif info.lane_index == center_lane:
#             assign_neighbors(neighbors_middle_idx, targets_middle, relative_pos, idx)
#         elif info.lane_index == center_lane - 1:
#             assign_neighbors(neighbors_down_idx, targets_down, relative_pos, idx)
#     for i in range(3):
#         idx_up = neighbors_up_idx[i]
#         idx_down = neighbors_down_idx[i]
#         # relative pos
#         if idx_up != -1:
#             neighbors_up[i, :2] = neighborhood_vehicle_states[idx_up][1][:-1] - ego_vehicle_state[1][:-1]
#             neighbors_up[i, 2] = float(neighborhood_vehicle_states[idx_up][3] - ego_vehicle_state[3])
#             neighbors_up[i, 3] = float(neighborhood_vehicle_states[idx_up][4] - ego_vehicle_state[4])
#         if idx_down != -1:
#             neighbors_down[i, :2] = neighborhood_vehicle_states[idx_down][1][:-1] - ego_vehicle_state[1][:-1]
#             # relative heading
#             neighbors_down[i, 2] = float(neighborhood_vehicle_states[idx_down][3] - ego_vehicle_state[3])
#             # relative speed
#             neighbors_down[i, 3] = float(neighborhood_vehicle_states[idx_down][4] - ego_vehicle_state[4])
#     for i in range(3):
#         if i != 1:
#             idx = neighbors_middle_idx[i]
#             if idx != -1:
#                 neighbors_middle[i, :2] = neighborhood_vehicle_states[idx][1][:-1] - ego_vehicle_state[1][:-1]
#                 neighbors_middle[i, 2] = float(neighborhood_vehicle_states[idx][3] - ego_vehicle_state[3])
#                 neighbors_middle[i, 3] = float(neighborhood_vehicle_states[idx][4] - ego_vehicle_state[4])
#     neighbors_middle = np.delete(neighbors_middle, 1, axis=0)
#     flatten_up = neighbors_up.flatten()
#     flatten_middle = neighbors_middle.flatten()
#     flatten_down = neighbors_down.flatten()
#     ego_v = np.zeros(13)
#     if len(obs.events.collisions) != 0:
#         ego_v[0] = 1
#     ego_v[1] = obs.events.off_road
#     ego_v[2] = obs.events.on_shoulder
#     # pos
#     ego_v[3:5] = ego_vehicle_state[1][:-1]
#     # heading
#     ego_v[5] = ego_vehicle_state[3]
#     # speed
#     ego_v[6] = ego_vehicle_state[4]
#     # linear speed
#     ego_v[7:13] = np.concatenate((ego_vehicle_state[11][:-1], ego_vehicle_state[12][:-1], ego_vehicle_state[13][:-1]))
#     obs_vectors = np.concatenate((flatten_up, flatten_middle, flatten_down, ego_v))
#     return obs_vectors
