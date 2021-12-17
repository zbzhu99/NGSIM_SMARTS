from psgail import *
from utils_psgail import *
import pickle as pk
import os
import time

def train(psgail, experts, stage, i_episode_res, num_episode=1000, print_every=10, gamma=0.95, batch_size=10000,
          agent_num=10, mini_epoch=3):
    rewards_log = []
    episodes_log = []
    env_creator = lambda: MATrafficSim(["./ngsim"], agent_number=agent_num)
    vector_env = ParallelEnv([env_creator] * 12, auto_reset=True)
    for i_episode in range(i_episode_res, num_episode):
        if i_episode % 200 == 0 and i_episode != 0:
            agent_num += 10
            vector_env.close()
            env_creator = lambda: MATrafficSim(["./ngsim"], agent_number=agent_num)
            vector_env = ParallelEnv([env_creator] * 12, auto_reset=True)
        if i_episode % 50 == 0 and i_episode != 0:
            with open('./models/psgail_' + str(int(i_episode / 50)) + '_' + stage + '.model',
                      "wb") as f:
                pk.dump(
                    {
                        'model': psgail,
                        'epoch': i_episode,
                        'rewards_log': rewards_log,
                        'episodes_log': episodes_log,
                        'agent_num': agent_num,
                        'stage': stage,
                    },
                    f,
                )
        time1 = time.time()
        states, next_states, actions, probs, dones, rewards = sampling(psgail, vector_env, batch_size=batch_size)
        time2 = time.time()
        print('sampling complete, time cost', time2-time1, 's')
        rewards_log.append(np.sum(rewards))
        episodes_log.append(i_episode)
        batch = trans2tensor({"state": states, "action": actions,
                              "probs": probs,
                              "next_state": next_states, "done": dones,
                              "reward": rewards, })
        # batch["adv"] = psgail.compute_adv(batch, gamma)
        experts_sample = np.random.randint(0, high=len(experts), size=len(states))
        cur_experts = experts[experts_sample]
        for i in range(mini_epoch):
            psgail.update_parameters(batch, cur_experts, gamma)
        if (i_episode + 1) % print_every == 0 or i_episode + 1 == num_episode:
            print("Episode: {}, Reward: {}".format(i_episode + 1, np.mean(rewards_log[-10:])))
    infos = {
        "rewards": rewards_log,
        "episodes": episodes_log
    }
    vector_env.close()
    return infos


if __name__ == "__main__":
    env_name = 'NGSIM SMARTS'
    train_episodes = 1000
    tuning_episodes = 200
    psgail = PSGAIL(discriminator_lr=5e-5,
                    policy_lr=1e-3,
                    value_lr=1e-3)
    experts = np.load('experts.npy')

    infos_1 = train(psgail, experts, 'train', 0, num_episode=train_episodes, print_every=100, gamma=0.95,
                    batch_size=40000,
                    agent_num=10, mini_epoch=3)

    plt.title('Reinforce training on {}'.format(env_name))
    plt.ylabel("Reward")
    plt.xlabel("Frame")
    infos = [infos_1, ]
    labels = ["PS-GAIL", ]
    for info in infos:
        x, y = info["episodes"], info["rewards"]
        y, x = moving_average(y, x)
        plt.plot(x, y)
    plt.legend(labels)
    plt.show()

    # fine tuning
    infos_2 = train(psgail, experts, 'tuning', 0, num_episode=tuning_episodes, print_every=10, gamma=0.99,
                    batch_size=40000,
                    agent_num=100, mini_epoch=1)

    plt.title('Reinforce tuning on {}'.format(env_name))
    plt.ylabel("Reward")
    plt.xlabel("Frame")
    infos = [infos_2, ]
    labels = ["PS-GAIL", ]
    for info in infos:
        x, y = info["episodes"], info["rewards"]
        y, x = moving_average(y, x)
        plt.plot(x, y)
    plt.legend(labels)
    plt.show()
