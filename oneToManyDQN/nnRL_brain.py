import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from multiRadar.env import Radar

env = Radar(450000,300)

# torch.manual_seed(1)


class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 50)
        # self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(50, 50)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        out = self.out(x)
        return out


class DeepQNetwork2(Radar):
    def __init__(self, n_states, n_actions):
        print("<DQN init>")
        # DQN有两个net:target net和eval net,具有选动作，存经历，学习三个基本功能
        self.eval_net, self.target_net = Net(n_states, n_actions), Net(n_states, n_actions)
        self.loss = nn.MSELoss()
        # lr = 0.0003
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.0003)
        self.n_actions = n_actions
        self.n_states = n_states
        # 使用变量
        self.learn_step_counter = 0  # target网络学习计数
        self.memory_counter = 0  # 记忆计数
        self.memory = np.zeros((300000, 5 * 2 + 2))  # 4*2+3
        self.cost = []  # 记录损失值
        self.which_action = []
    def choose_action(self, x, epsilon):
        u = random.random()
        if u < epsilon:
            # forward feed the observation and get q value for every actions
            s = torch.FloatTensor(x)
            actions_value = self.eval_net(s)
            # actions_value = actions_value.reshape([3, 40])
            print("value：")
            print(actions_value)
            value_max = np.max(actions_value.detach().numpy())
            print("value表中最大值：",value_max)
            action  = np.argmax(actions_value.detach().numpy()) #最大值所在的位置
            # self.which_action.append(env.action_space[action][0])
        if u >= epsilon:
            # action = [np.array([np.random.randint(0, 3)]), np.array([np.random.randint(0, 40)])]
            action = np.random.randint(0,self.n_actions)
        return action

    def store_transition(self, state, action, reward, next_state):
        # print("<store_transition>")
        transition = np.hstack((state, action, reward, next_state))
        index = self.memory_counter % 300000  # 满了就覆盖旧的
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # print("<learn>")
        # target net 更新频率,用于预测，不会及时更新参数
        if self.learn_step_counter % 1000 == 0:
            self.target_net.load_state_dict((self.eval_net.state_dict()))
            # print('\ntarget_params_replaced\n')
        self.learn_step_counter += 1

        # 使用记忆库中批量数据
        sample_index = np.random.choice(300000, 32)  # 300000个中随机抽取32个作为batch_size
        memory = self.memory[sample_index, :]  # 抽取的记忆单元，并逐个提取
        state = torch.FloatTensor(memory[:, :5])
        action = torch.LongTensor(memory[:, 5:6])
        reward = torch.LongTensor(memory[:, 6:7])
        next_state = torch.FloatTensor(memory[:, 7:12])

        # 计算loss,q_eval:所采取动作的预测value,q_target:所采取动作的实际value
        q_eval = self.eval_net(state).gather(1, action)  # eval_net->(64,4)->按照action索引提取出q_value
        q_next = self.target_net(next_state).detach()
        # torch.max->[values=[],indices=[]] max(1)[0]->values=[]
        q_target = reward + 0.9 * q_next.max(1)[0].unsqueeze(1)  # label 0.99
        loss = self.loss(q_eval, q_target)
        loss_ = loss.detach()
        self.cost.append(loss_)  # 切断反向传播
        # 反向传播更新
        self.optimizer.zero_grad()  # 梯度重置
        loss.backward()  # 反向求导
        self.optimizer.step()  # 更新模型参数

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost)), self.cost)
        plt.xlabel("step")
        plt.ylabel("Cost")
        plt.savefig('cost.png')
        plt.show()
