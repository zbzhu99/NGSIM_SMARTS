import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


class Net(torch.nn.Module):
    def __init__(self, hidden_size, input_size, output_size, layer_num=2):
        super(Net, self).__init__()
        # Use torch.nn.ModuleList or torch.nn.Sequential For multiple layers
        layers = []
        last_size = input_size
        for i in range(layer_num - 1):
            layers.append(torch.nn.Linear(last_size, hidden_size))
            layers.append(torch.nn.ReLU())
            last_size = hidden_size
        layers.append(torch.nn.Linear(last_size, output_size))
        self._net = torch.nn.Sequential(*layers)
        self._net.to(device)

    def forward(self, inputs):
        res = self._net(inputs)
        # res = torch.sigmoid(res)
        return res


class NetBN(torch.nn.Module):
    def __init__(self, hidden_size, input_size, output_size, layer_num=2):
        super(NetBN, self).__init__()
        # Use torch.nn.ModuleList or torch.nn.Sequential For multiple layers
        layers = []
        last_size = input_size
        for i in range(layer_num - 1):
            layers.append(torch.nn.Linear(last_size, hidden_size))
            layers.append(torch.nn.BatchNorm1d(hidden_size))
            layers.append(torch.nn.ReLU())
            last_size = hidden_size
        layers.append(torch.nn.Linear(last_size, output_size))
        self._net = torch.nn.Sequential(*layers)
        self._net.to(device)

    def forward(self, inputs):
        res = self._net(inputs)
        # res = torch.sigmoid(res)
        return res


# class GRUNet(torch.nn.Module):
#     def __init__(self, state_space, gru_hidden, gru_nums):
#         self.grus = nn.GRU(state_space, gru_hidden, gru_nums, bias=False, dropout=0, batch_first=True)
#         self.grus.to(device)
#
#     def forward(self, inputs, hidden=None):
#         return self.grus(inputs, hidden)


class PSGAIL():
    def __init__(self,
                 discriminator_lr,
                 policy_lr,
                 value_lr,
                 hidden_size=128,
                 state_action_space=38,
                 state_space=36,
                 gru_nums=64,
                 gru_hidden=4,
                 ):
        self._tau = 0.1
        self._clip_range = 0.2
        self.lambda_gp = 4
        self.discriminator = Net(hidden_size, state_action_space, output_size=1)
        self.value = NetBN(hidden_size, state_space, output_size=1)
        self.target_value = NetBN(hidden_size, state_space, output_size=1)
        self.policy = NetBN(hidden_size, state_space, output_size=gru_hidden, layer_num=3)
        # self.policy = GRUNet(state_space, gru_hidden, gru_nums)
        self.discriminator_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=discriminator_lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        # self.last_hidden = None

    def get_r(self, obs_action):
        return self.discriminator(obs_action)

    # def get_action(self, obs, hidden=None, action=None):
    #     policy_out, self.last_hidden = self.policy(obs, hidden)
    #     mean1, var1, mean2, var2 = torch.chunk(policy_out, 4, dim=-1)
    #     mean1 = 2 * torch.tanh(mean1)
    #     var1 = torch.nn.functional.softplus(var1)
    #     mean2 = 2 * torch.tanh(mean2)
    #     var2 = torch.nn.functional.softplus(var2)
    #     m1 = Normal(mean1, var1)
    #     m2 = Normal(mean2, var2)
    #     if action is None:
    #         action1 = m1.sample()
    #         action2 = m2.sample()
    #         action1 = torch.clamp(action1, -2, 2)
    #         action2 = torch.clamp(action2, -2, 2)
    #         log_prob1 = m1.log_prob(action1)
    #         log_prob2 = m2.log_prob(action2)
    #         action = torch.concat((action1, action2), dim=1)
    #     else:
    #         log_prob1 = m1.log_prob(action[:, 0])
    #         log_prob2 = m2.log_prob(action[:, 1])
    #     log_prob = log_prob2 + log_prob1
    #     prob = torch.exp(log_prob)
    #     return log_prob.reshape(-1, 1), prob.reshape(-1, 1), action, self.last_hidden

    def get_action(self, obs, action=None):
        policy_out = self.policy(obs)
        mean1, var1, mean2, var2 = torch.chunk(policy_out, 4, dim=-1)
        mean1 = 2 * torch.tanh(mean1)
        var1 = torch.nn.functional.softplus(var1)
        mean2 = 2 * torch.tanh(mean2)
        var2 = torch.nn.functional.softplus(var2)
        m1 = Normal(mean1, var1)
        m2 = Normal(mean2, var2)
        if action is None:
            action1 = m1.sample()
            action2 = m2.sample()
            log_prob1 = m1.log_prob(action1)
            log_prob2 = m2.log_prob(action2)
            action1 = torch.clamp(action1, -2, 2)
            action2 = torch.clamp(action2, -2, 2)
            action = torch.cat((action1, action2), dim=1)
        else:
            action1 = action[:, 0].unsqueeze(1)
            action2 = action[:, 1].unsqueeze(1)
            log_prob1 = m1.log_prob(action1)
            log_prob2 = m2.log_prob(action2)
        log_prob = log_prob2 + log_prob1
        prob = torch.exp(log_prob.squeeze()).unsqueeze(1)
        return log_prob.reshape(-1, 1), prob.reshape(-1, 1), action

    def grad_penalty(self, agent_data, experts_data):
        alpha = torch.tensor(np.random.random(size=experts_data.shape), dtype=torch.float32).cuda()
        interpolates = (alpha * experts_data + ((1 - alpha) * agent_data)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(d_interpolates.size()).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = self.lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def soft_update(self, source, target, tau=None):
        if tau is None:
            tau = self._tau
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update_parameters(self, batch, sap_experts, gamma):
        s = batch["state"]
        a = batch["action"]
        s1 = batch["next_state"]
        done = batch["done"].reshape(-1, 1)
        old_prob = batch["probs"].reshape(-1, 1)

        sap_experts = torch.tensor(sap_experts, device=device, dtype=torch.float32)
        sap_agents = torch.cat((s, a), dim=1)
        sap_agents = sap_agents.detach()
        D_expert = self.discriminator(sap_experts)
        D_agents = self.discriminator(sap_agents)
        grad_penalty = self.grad_penalty(sap_agents.data, sap_experts.data)
        discriminator_loss = D_agents.mean() - D_expert.mean() + grad_penalty
        self.discriminator_optimizer.zero_grad()
        # discriminator_loss.backward(retain_graph=True)
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        agents_rew = (D_agents - D_agents.mean()) / (D_agents.std() + 1e-8)
        td_target = agents_rew.detach() + gamma * self.target_value(s1) * (1 - done)
        td_delta = td_target - self.value(s)
        log_prob, cur_prob, action = self.get_action(s, a)
        ip_sp = cur_prob / (old_prob + 1e-7)
        ip_sp_clip = torch.clamp(ip_sp, 1 - self._clip_range, 1 + self._clip_range)
        policy_loss = -torch.mean(torch.min(ip_sp * td_delta.detach(), ip_sp_clip * td_delta.detach()))
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        value_loss = torch.mean(F.mse_loss(self.value(s), td_target.detach()))
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.soft_update(self.value, self.target_value, self._tau)
