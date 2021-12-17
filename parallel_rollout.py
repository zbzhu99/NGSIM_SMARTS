import sys
import numpy as np

from multiagent_traffic_simulator import MATrafficSim
from smarts.env.wrappers.parallel_env import ParallelEnv
from utils_psgail import obs_extractor
import random
import datetime
from collections import defaultdict

# Increase system recursion limit
sys.setrecursionlimit(2500000)
random.seed(datetime.datetime.now())


def getlist(list_, idx):
    if idx < 0 or idx >= len(list_) or len(list_) == 0:
        return None
    else:
        return list_[idx]


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


# def dump_trajectory(expert_trajectory, agent_id, batch_samples):
#     batch_samples.states += expert_trajectory[agent_id].states
#     batch_samples.probs += expert_trajectory[agent_id].probs
#     batch_samples.actions += expert_trajectory[agent_id].actions
#     batch_samples.hiddens += expert_trajectory[agent_id].hiddens
#     batch_samples.next_states += expert_trajectory[agent_id].next_states
#     batch_samples.rewards += expert_trajectory[agent_id].rewards
#     batch_samples.dones += expert_trajectory[agent_id].dones

def dump_trajectory(expert_trajectory, batch_samples):
    batch_samples.states += expert_trajectory.states
    batch_samples.probs += expert_trajectory.probs
    batch_samples.actions += expert_trajectory.actions
    batch_samples.hiddens += expert_trajectory.hiddens
    batch_samples.next_states += expert_trajectory.next_states
    batch_samples.rewards += expert_trajectory.rewards
    batch_samples.dones += expert_trajectory.dones


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


if __name__ == "__main__":
    env_num = 12
    agent_number = 10
    env_creator = lambda: MATrafficSim(["./ngsim"], agent_number=agent_number)
    vector_env = ParallelEnv([env_creator] * env_num, auto_reset=True)
    vector_env.seed(random.randint(1, 500))
    vec_obs = vector_env.reset()
    sap_size = 16384
    batch_size = 64
    trajec_dict = defaultdict(trajectory)
    vec_done = []
    expert_trajectory = {}
    batch_samples = defaultdict(samples_agents)
    hidden = None
    for i in range(env_num):
        expert_trajectory[i] = {}
    counter = 0
    while True:
        vec_act = []
        vec_prob = []
        obs_vectors = np.zeros((1, 36))
        for idx, obs in enumerate(vec_obs):
            act_n = {}
            for agent_id in obs.keys():
                if agent_id not in expert_trajectory[idx]:
                    expert_trajectory[idx][agent_id] = trajectory()
                elif vec_done[idx].get(agent_id) is None or vec_done[idx].get(agent_id):
                    continue
                # obs_vectors = obs_extractor(obs[agent_id])
                obs_vectors = np.vstack((obs_vectors, obs_extractor(obs[agent_id])))
                expert_trajectory[idx][agent_id].states.append(obs_vectors[-1, :])
                act_n[agent_id] = np.random.normal(0, 1, size=(2,))
                prob = np.random.normal(0, 1, size=(1,))
                hidden = np.random.normal(0, 1, size=(1,))
                expert_trajectory[idx][agent_id].probs.append(prob)
                expert_trajectory[idx][agent_id].actions.append(act_n[agent_id])
                expert_trajectory[idx][agent_id].hiddens.append(hidden)
            vec_act.append(act_n)
        # hidden = np.delete(hidden, del_list, 0)
        # log_prob, prob, acts, hidden = psgail.get_action(obs_vectors, hidden)
        # act_idx = 0
        # for experts in expert_trajectory.values():
        #     act_n = {}
        #     for agent_id in experts:
        #         act_idx += 1
        #         act_n[agent_id] = acts[act_idx]
        #         expert_trajectory[idx][agent_id].probs.append(prob[act_idx])
        #         expert_trajectory[idx][agent_id].hiddens.append(hidden[act_idx])
        #         expert_trajectory[idx][agent_id].actions.append(act_n[agent_id])
        vec_obs, vec_rew, vec_done, vec_info = vector_env.step(vec_act)
        # del_idx = 0
        # del_list = []
        for idx, act_n in enumerate(vec_act):
            for agent_id in act_n.keys():
                obs_vectors = obs_extractor(vec_obs[idx].get(agent_id))
                expert_trajectory[idx][agent_id].next_states.append(obs_vectors)
                expert_trajectory[idx][agent_id].rewards.append(vec_rew[idx].get(agent_id))
                expert_trajectory[idx][agent_id].dones.append(vec_done[idx].get(agent_id))
                # del_idx += 1
                counter += 1
                # if expert_trajectory[idx][agent_id].dones[-1]:
                #     # del_list.append(del_idx)
                #     dump_trajectory(expert_trajectory[idx], agent_id, batch_samples)
        if counter >= sap_size:
            # dump_all(expert_trajectory, batch_samples)
            break
    # len_sum = 0
    # idx_traj = 0
    # for idx, traj in enumerate(expert_trajectory.values()):
    #     for agent_traj in traj.values():
    #         trajec_dict[idx_traj] = agent_traj
    #         idx_traj += 1
    #         len_sum += len(agent_traj.dones)
    avg_num = int(counter / batch_size)
    # trajec_list = sorted(trajec_dict.items(), key=lambda item: len(item[1].dones), reverse=True)
    for idx, traj in enumerate(expert_trajectory.values()):
        for agent_traj in traj.values():
            cur_len = len(agent_traj.dones)
            min = 10000
            min_sec = 5000
            app = -1
            app_sec = -1
            for i in range(batch_size):
                dis = len(batch_samples[i].dones) + cur_len - avg_num
                if dis <= 0:
                    if abs(dis) < min:
                        min = abs(dis)
                        app = i
                elif abs(dis) < min_sec:
                    min_sec = abs(dis)
                    app_sec = i
            if app != -1:
                dump_trajectory(agent_traj, batch_samples[app])
            elif app_sec != -1:
                dump_trajectory(agent_traj, batch_samples[app_sec])

    # for tup in trajec_list:
    #     cur_len = len(tup[1].dones)
    #     min = 10000
    #     # min_sec = 5000
    #     app = -1
    #     # app_sec = -1
    #     for i in range(batch_size):
    #         dis = len(batch_samples[i].dones) + cur_len - avg_num
    #         if abs(dis) < min:
    #             min = abs(dis)
    #             app = i
    #     dump_trajectory(tup[1], batch_samples[app])

    vector_env.close()

# if __name__ == "__main__":
#     env_num = 12
#     env_creator = lambda: MATrafficSim(["./ngsim"], agent_number=10)
#     vector_env = ParallelEnv([env_creator] * env_num, auto_reset=True)
#     vector_env.seed(random.randint(1, 500))
#     vec_obs = vector_env.reset()
#     sap_size = 5000
#     vec_done = []
#     expert_trajectory = {}
#     batch_samples = samples_agents()
#     for i in range(env_num):
#         expert_trajectory[i] = {}
#     counter = 0
#     while True:
#         vec_act = []
#         vec_prob = []
#         for idx, obs in enumerate(vec_obs):
#             act_n = {}
#             for agent_id in obs.keys():
#                 if agent_id not in expert_trajectory[idx]:
#                     expert_trajectory[idx][agent_id] = trajectory()
#                 elif vec_done[idx][agent_id]:
#                     del expert_trajectory[idx][agent_id]
#                     continue
#                 obs_vectors = obs_extractor(obs[agent_id])
#                 act_n[agent_id] = np.random.normal(0, 1, size=(2,))
#                 prob = np.random.normal(0, 1, size=(1,))
#                 hidden = np.random.normal(0, 1, size=(1,))
#                 expert_trajectory[idx][agent_id].states.append(obs_vectors)
#                 expert_trajectory[idx][agent_id].probs.append(prob)
#                 expert_trajectory[idx][agent_id].actions.append(act_n[agent_id])
#                 expert_trajectory[idx][agent_id].hiddens.append(hidden)
#             vec_act.append(act_n)
#         vec_obs, vec_rew, vec_done, vec_info = vector_env.step(vec_act)
#         for idx, act_n in enumerate(vec_act):
#             for agent_id in act_n.keys():
#                 obs_vectors = obs_extractor(vec_obs[idx].get(agent_id))
#                 expert_trajectory[idx][agent_id].next_states.append(obs_vectors)
#                 expert_trajectory[idx][agent_id].rewards.append(vec_rew[idx].get(agent_id))
#                 expert_trajectory[idx][agent_id].dones.append(vec_done[idx].get(agent_id))
#                 counter += 1
#                 if expert_trajectory[idx][agent_id].dones[-1]:
#                     dump_trajectory(expert_trajectory[idx], agent_id, batch_samples)
#         if counter >= sap_size:
#             dump_all(expert_trajectory, batch_samples)
#             break
#     vector_env.close()
