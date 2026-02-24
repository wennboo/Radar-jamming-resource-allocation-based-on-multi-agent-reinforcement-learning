import numpy as np
import time
import math


class Radar():
    def __init__(self, distance, v):
        # 雷达的初始状态为搜索
        self.distance = distance
        self.v = v
        self.state = np.array([0, 0, 0, self.distance,100])
        # 计数器：搜索状态时，被探测到的次数
        self.search_detect_count = [[], [], []]
        # 计数器：跟踪状态时，被探测到的次数
        self.track_detect_count = [[], [], []]
        self.reward = 0
        self.sum_reward = 0
        self.done = False
        self.pj = 100


    def reset(self):
        # 重置雷达状态为搜索
        self.state = np.array([0, 0, 0, self.distance,100])  # [雷达1状态，雷达2状态,雷达3状态]
        self.search_detect_count = [[], [], []]
        self.track_detect_count = [[], [], []]
        self.reward = 0
        self.sum_reward = 0
        self.done = False
        self.pj = 100

        return self.state

    def planePosition(self, t):  # 干扰机到雷达2的径向距离
        L = self.distance - self.v * t
        return L

    def radial(self, t, action):  # 干扰机到雷达的径向距离
        L = self.distance - self.v * t
        if action < 3 or action > 5:
            R = math.sqrt(L ** 2 + 5000 ** 2)
        else:
            R = L
        return R

    def jamPw(self, action):
        perPw = 100
        if action == 0 or action == 3 or action == 6:
            print("加功率")
            self.pj += perPw
        elif action == 1 or action == 4 or action == 7:
            print("减功率")
            self.pj -= perPw
        else:
            print("功率不变")
            self.pj = self.pj
        if self.pj <= 100:
            self.pj = 100
        if self.pj >= 3000:
            self.pj = 3000
        # Pj = self.action_space[action][1]
        print("当前施加的干扰功率为：", self.pj)
        return self.pj

    def radarFind(self, R, action, pw):  # 雷达当前的最大探测距离
        pt = 100000
        gt = 10 ** (42 / 10)
        gr = 10 ** (42 / 10)
        lamada = 0.0375
        sigma = 6
        # D = 1 / 6.4e-6
        D = 34000
        k = 1.38e-23
        t0 = 290
        bn = 30000000
        fn = 10 ** (3 / 10)
        pc = 10 ** (36 / 10)
        n = 10
        gj = 10 ** (15 / 10)
        gt_theta = 10 ** (15 / 10)
        gamma_j = 10 ** (0.5 / 10)
        Lt = 10 ** (6 / 10)
        # dsn = 10 ** (11 / 10)
        dsn = 10
        if action < 3 or action > 5:
            arcsin = math.asin(5000 / R)  # 弧度
            angel = arcsin * 180 / np.pi  # 角度
            if angel >= 0 and angel <= 3:
                gt_theta = gt_theta
            if angel > 3 and angel <= 90:
                gt_theta = 0.04 * gt_theta * (6 / angel) ** 2
            if angel > 90 and angel <= 180:
                gt_theta = 0.04 * gt_theta * (6 / 90) ** 2

        find = (pt * gt * gr * (lamada ** 2) * sigma * D) / ((4 * np.pi) ** 3 * (
                (k * t0 * bn * fn) + (pw * gj * gt_theta * (lamada ** 2) * gamma_j) / (
                (4 * np.pi) ** 2 * R ** 2 + 0.000001)) * dsn * Lt)
        find = find ** 0.25
        print("此时雷达的探测距离为：", find)

        return find

    def patternChange(self, t, action, index, pw):
        # 当前选择的雷达编号
        # index = action[0][0]
        print("当前的雷达是：", index)
        if self.state[index] == 0:  # 如果当前雷达处于搜索状态
            # 如果干扰机的位置在此时的探测范围内
            if self.radial(t, action) \
                    <= self.radarFind(self.radial(t, action), action, pw):
                # 雷达的搜索计数器+1
                self.search_detect_count[index].append(1)
            else:
                self.search_detect_count[index].append(0)
            # 搜索状态探测4次
            if len(self.search_detect_count[index]) == 4:
                # 如果干扰机被探测到3次
                if sum(self.search_detect_count[index]) >= 3:
                    # 那么当前雷达的状态由搜索转变为跟踪
                    self.state[index] = 1
                    self.search_detect_count[index] = []
                    self.reward -= 5
                else:  # 保持搜索状态不变
                    self.search_detect_count[index].pop(0)
                    self.reward += 1

        if self.state[index] == 1:  # 如果当前雷达处于跟踪状态
            if self.radial(t, action) \
                    <= self.radarFind(self.radial(t, action), action, pw):
                # 雷达的跟踪计数器+1
                self.track_detect_count[index].append(1)
            else:
                self.track_detect_count[index].append(0)
                # 跟踪状态探测3次
            if len(self.track_detect_count[index]) == 3:
                # 如果干扰机被探测到2次
                if sum(self.track_detect_count[index]) >= 2:
                    # 那么当前的雷达状态由跟踪转变为制导
                    self.state[index] = 2
                    self.track_detect_count[index] = []
                    self.reward -= 5
                    self.done = True
                elif sum(self.track_detect_count[index]) == 1:
                    # 那么当前的雷达状态保持不变
                    self.track_detect_count[index].pop(0)
                    self.reward += 1
                else:  # 那么当前的雷达状态由跟踪转变为搜索
                    self.state[index] = 0
                    self.track_detect_count[index] = []
                    self.reward += 5

    # def pjReward(self, action):
    #     if action == 0 or action == 3 or action == 6:
    #         rj = -0.5
    #     elif action == 1 or action == 4 or action == 7:
    #         rj = 5
    #     else:
    #         rj = -0.1
    #     return rj

    def step(self, action, t):  # 0-2,0-39，1500秒，v=300m/s
        # 选择雷达
        if 0 <= action <= 2:
            index = 0
        elif 3 <= action <= 5:
            index = 1
        else:
            index = 2
        pw = self.jamPw(action)
        self.patternChange(t, action, index, pw)
        # rj = self.pjReward(action)
        # print("rj为：",rj)
        # self.reward += rj
        print("reward为：",self.reward)
        if index == 0:
            index = 1
            self.patternChange(t, action, index, pw)
            index = 2
            self.patternChange(t, action, index, pw)
            index = 0
        elif index == 1:
            index = 0
            self.patternChange(t, action, index, pw)
            index = 2
            self.patternChange(t, action, index, pw)
            index = 1
        else:
            index = 0
            self.patternChange(t, action, index, pw)
            index = 1
            self.patternChange(t, action, index, pw)
            index = 2

        self.sum_reward += self.reward
        self.state[3] = self.planePosition(t)
        self.state[4] = self.pj

        return self.state, self.reward, self.done
