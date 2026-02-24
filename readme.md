本项目运行平台：操作系统Windows11，处理器intel i512500h。训练最简单的情况大约耗时1小时。

项目依赖：
	gym		     0.26.1
	gymnasium	0.29.1
	matplotlib 	  3.4.3
	numpy		1.21.0
	python 		3.9.7
	torch		2.2.1



环境编写函数说明：

```
step(self, actions, t, task_dict)
输入：干扰动作集合，当前步数，任务分配集合
输出：状态集合、累积奖励、回合是否结束标识

# 雷达工作模式转换函数
patternChange(self, t, index, pj_type, pj_change, task_dict, flag)
输入：当前步数，雷达编号，干扰样式，干扰功率，任务分配集合，位置计算标识
输出：雷达工作模式，奖励累积

# 奖励函数设计
calRadarReward(self, index)
输入：当前雷达状态
输出：根据雷达威胁等级所获得的奖励
calTypeReward(self, pj_type, index, t, task_dict)
输入：干扰样式，雷达编号，当前步数，任务分配集合
输出：干扰能量最小化奖励
callPowerReward(self, pj_change)
输入：干扰动作中的功率编号
输出：干扰功率奖励

#选择功率
jamPj(self, pj_type, pj_change)
输入：干扰样式，干扰动作中的功率编号
输出：干扰功率

#干扰任务分配函数
taskArrange(self, radar_state_dict)
输入：雷达工作模式集合
输出：分配结果集合
```