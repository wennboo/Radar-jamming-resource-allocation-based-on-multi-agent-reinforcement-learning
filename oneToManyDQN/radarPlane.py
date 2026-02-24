from copy import deepcopy
import numpy as np
import torch
from matplotlib import pyplot as plt
from new.env import Radar
# from multiRadar.RL_DQN import *
from nnRL_brain import DeepQNetwork2
from nnRL_brain import Net

env = Radar(distance=450000, v=300)
# 三个雷达，三个状态,一个距离
state_size = 5
# 三个雷达，四种干扰，十个功率
action_size = 9
num_episodes = 1000
MEANS_REWARD = 200
MEANS_STEP = 200
reward_arr = []
step_arr = []
avg_reward = []
avg_step = []
per_pj = []
per_action = []
agent = DeepQNetwork2(n_states=state_size, n_actions=action_size)
epsilon = 0


for episode in range(num_episodes):

    step = 1
    observation = env.reset()
    done = False
    # 0.000083
    epsilon = epsilon + 1 / num_episodes if epsilon < 0.9 else 0.9

    for t in range(1, int(env.distance / env.v + 1)):
        env.reward = 0
        print("==============================================================")
        action = agent.choose_action(observation, epsilon)
        print("选取的动作为：", action)
        per_action.append(action)
        # print("最大值在value表中的位置：",action)
        # print(action[1][0])
        next_obs, rewards, done = env.step(action, t)
        print("雷达搜索计数器：", env.search_detect_count)
        print("雷达跟踪计数器：", env.track_detect_count)
        print("奖励：", rewards)
        print(done)
        print("当前时刻的状态：", observation)
        print("下一时刻的状态:", next_obs)
        print('step=', step)
        agent.store_transition(observation, action, rewards, next_obs)
        print("目前存储次数:", agent.memory_counter)
        observation = deepcopy(next_obs)
        # 超过200条每隔5步学习一次
        if agent.memory_counter > 200:
            agent.learn()
        print("此时飞机的位置：", env.planePosition(t))
        if env.planePosition(t) <= 0:
            done = True
        if done:
            print("================================突防结束========================================")
            break
        step += 1
        per_pj.append(env.pj)
        # print(per_pj)
    print("本轮的累积奖励为：", env.sum_reward)

    reward_arr.append(env.sum_reward)
    avg_r = np.mean(reward_arr[-MEANS_REWARD:])
    avg_reward.append(avg_r)

    step_arr.append(step)
    avg_s = np.mean(step_arr[-MEANS_STEP:])
    avg_step.append(avg_s)


# agent.plot_cost()
# 奖励图像
plt.figure()
plt.plot(np.arange(len(reward_arr)), reward_arr)
plt.savefig('reward.png')
plt.show()
# 平均奖励图像
plt.figure()
plt.plot(np.arange(len(avg_reward)), avg_reward)
plt.savefig('avg_reward.png')
plt.show()
# 突防步数图像
plt.figure()
plt.plot(np.arange(len(step_arr)), step_arr)
plt.savefig('step.png')
plt.show()
# 平均步数图像
np.savetxt('DQN_data.txt',avg_step)
plt.figure()
plt.plot(np.arange(len(avg_step)), avg_step)
plt.savefig('avg_step.png')
plt.show()

plt.figure()
plt.plot(np.arange(len(per_pj)), per_pj)
plt.savefig('per_pj.png')
plt.show()

plt.figure()
plt.plot(np.arange(len(per_action)), per_action)
plt.savefig('per_action.png')
plt.show()

# plt.figure()
# plt.plot(np.arange(len(agent.which_action)), agent.which_action)
# plt.savefig('which_action')
# plt.show()
