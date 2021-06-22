import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from queue import Queue
import pandas as pd

G = 9.8
UNREASONABLE_DIS = 95.0


class CarChangeLaneEnv(gym.Env):

    def __init__(self):

        self.seed()
        self.time_step = 0.02  # 时间记得根据仿真环境后续进行调整
        self.steps_beyond_done = None
        self.isReady = False
        self.isReset = False
        self.isWarmUp = True
        self.epiNum = 0
        self.trainReward = Queue(25)
        self.evalReward = Queue(7)
        self.isRecord = False
        self.x = 0
        self.y = 0
        self.traceRecord = pd.DataFrame([[0, 0]], columns=['x', 'y'])
        # —————————————————————————————————————————————lstm轨迹加在这里——-----------——------------------------------------
        self.lstmTrace = pd.read_csv('LSTM_Trace/290_LSTM.csv')
        self.index = 1
        self.overIndex = 1

        # 强化学习状态量和动作量
        self.init_env_state = None
        self.init_env_action = None
        self.state = None
        self.action = None
        # —————————————————————————————————————————————待解决(1): 需要定义状态及动作的范围------------------------------------
        state_high_limit = np.array([48.91, 49.33, 102.2, 6.67, 4.60, 18.9, 6.69, 6.69, 12.93, 68.3, 25.2])   # 状态量的上限，（维度与状态相同的向量）
        state_low_limit = np.array([0, 0, 0, 0, 0, 0, 0, -9.4, -8.5, 0, 0])    # 状态量的下限，（维度与状态相同的向量）
        action_high_limit = np.array([3.4, 1.4])  # 动作量的上限，（维度与动作相同的向量）
        action_low_limit = np.array([-1, -1])  # 动作量的下限，（维度与动作相同的向量）
        self.observation_space = spaces.Box(low=state_low_limit, high=state_high_limit)
        self.action_space = spaces.Box(low=action_low_limit,  high=action_high_limit)
        self.obsQueue = Queue(maxsize=10000)    # 用于存储观测值
        self.actQueue = Queue(maxsize=10000)     # 用于存储动作值

        # 环境自身状态量
        self.isCrashed = False
        self.isOver = False

        # Unity 通信变量
        self.v_pre_goal = 0.0
        self.v_tail_goal = 0.0
        self.v_pre_raw = 0.0
        self.y_pre_goal = 0.0
        self.y_tail_goal = 0.0
        self.y_pre_raw = 0.0

        self.y_pre_goal_old = 0.0
        self.y_tail_goal_old = 0.0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def updateState(self, action):
        action = action.squeeze()
        d_pre_goal, d_tail_goal, d_pre_raw, interval_pre, interval_tail, v_x, v_y, delta_v_goal, delta_v_raw, delta_a_x, delta_a_y = self.state
        _a_x, _a_y = self.action
        a_x, a_y = action
        self.action = action
        x_moved = v_x * self.time_step + 0.5 * a_x * self.time_step ** 2
        y_moved = v_y * self.time_step + 0.5 * a_y * self.time_step ** 2
        self.actQueue.put([x_moved / 1.5, y_moved / 1.5])
        if self.isRecord:
            self.x = self.x + x_moved / 1.5
            self.y = self.y + y_moved / 1.5
            print(self.x, self.y)
            self.traceRecord = self.traceRecord.append({'x': self.x, 'y': self.y}, ignore_index=True)
        # app.send_move_message({'z_moved': x_moved, 'x_moved': y_moved})

        d_pre_goal = d_pre_goal + self.v_pre_goal * self.time_step - x_moved
        d_tail_goal = d_tail_goal - self.v_tail_goal * self.time_step + x_moved
        d_pre_raw = d_pre_raw + self.v_pre_raw * self.time_step - x_moved

        if self.y_pre_goal != 0.0 and self.y_pre_goal_old == 0.0:
            self.y_pre_goal_old = self.y_pre_goal
        if self.y_tail_goal != 0.0 and self.y_tail_goal_old == 0.0:
            self.y_tail_goal_old = self.y_tail_goal
        interval_pre = interval_pre + self.y_pre_goal - self.y_pre_goal_old - y_moved
        interval_tail = interval_tail + self.y_tail_goal - self.y_tail_goal_old - y_moved
        self.y_pre_goal_old = self.y_pre_goal
        self.y_tail_goal_old = self.y_tail_goal

        v_x = v_x + a_x * self.time_step
        v_y = v_y + a_y * self.time_step

        delta_v_goal = v_x - self.v_pre_goal
        delta_v_raw = v_x - self.v_pre_raw
        delta_a_x = (a_x - _a_x) / self.time_step
        delta_a_y = (a_y - _a_y) / self.time_step

        # print('THE AUTOMATIC VEHICLE STEP IN WITH: {} {}\n'.format(x_moved / 1.5, y_moved / 1.5))  # 通知仿真环境移动（stepPhysics）
        self.state = [d_pre_goal, d_tail_goal, d_pre_raw, interval_pre, interval_tail, v_x, v_y, delta_v_goal, delta_v_raw, delta_a_x, delta_a_y]

        return d_pre_goal, d_tail_goal, d_pre_raw, interval_pre, interval_tail, v_x, v_y, delta_v_goal, delta_v_raw, delta_a_x, delta_a_y

    def step(self, action):
        # action = np.expand_dims(action, axis=0)
        action = action.squeeze()
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        d_pre_goal, d_tail_goal, d_pre_raw, interval_pre, interval_tail, v_x, v_y, delta_v_goal, delta_v_raw, delta_a_x, delta_a_y = self.updateState(action)
        # assert self.observation_space.contains(self.state), "%r (%s) invalid" % (self.state, type(self.state))

        # 判断碰撞
        if np.sqrt(d_pre_goal ** 2 + interval_pre ** 2) < 3:
            self.isCrashed = True
            crashTyp = 1
        elif np.sqrt(d_tail_goal ** 2 + interval_tail ** 2) < 3:
            self.isCrashed = True
            crashTyp = 2
        elif np.sqrt(d_pre_raw ** 2 + (abs(self.y_pre_raw - self.y_tail_goal) - interval_tail) ** 2) < 3:
            self.isCrashed = True
            crashTyp = 3

        # 判断换道结束
        if interval_tail <= 0 and interval_pre <= 0:
        #     self.overIndex = self.index
        #
        # if self.overIndex - self.index >= 10:
            self.isOver = True

        # 判断结束
        done = False
        if self.isCrashed or self.isOver:  # isOver：换道结束
            if self.isCrashed:
                print('CRASH DONE! CRASH TYPE: {}'.format(crashTyp))
            else:
                print('OVER DONE! ')
            done = True

        # —————————————————————待解决(2): 需要调整未换道惩罚值，以及isOver的情况（done=true导致返回的reward是0）------------------
        # 计算奖励值
        reward = 0.0
        TTC1 = d_pre_goal / (-1.0 * delta_v_goal)
        TTC2 = d_tail_goal / (v_x - self.v_tail_goal)
        TTC3 = d_pre_raw / (-1.0 * delta_v_raw)
        if 2/3 * abs(self.y_pre_raw - self.y_tail_goal) < abs(interval_tail) <= abs(self.y_pre_raw - self.y_tail_goal):
            S = TTC3 / 2220
            typ = 1
        elif 1/3 * abs(self.y_pre_raw - self.y_tail_goal) < abs(interval_tail) <= 2/3 * abs(self.y_pre_raw - self.y_tail_goal):
            S = 0.318 * (TTC1 + 60) / 2260 + 0.447 * (TTC2 + 4200) / 1130 + 0.209 * TTC3 / 2220
            typ = 2
        else:
            S = 0.402 * (TTC1 + 60) / 2260 + 0.598 * (TTC2 + 4200) / 1130
            typ = 3
        a_x, _ = self.action
        w = v_y / v_x
        C = -0.547 * (a_x + 3) / 8 - 0.099 * (delta_a_x + 70) / 140 - 0.209 * (w + 0.4) / 0.8
        E = v_x / 20.0
        bias = np.sqrt((self.x - self.lstmTrace.iloc[self.index, 1])**2 +
                       (self.y - self.lstmTrace.iloc[self.index, 0])**2) * 0.01
        # print(self.y - self.lstmTrace.iloc[self.index, 0])
        # print(self.index)
        self.index += 1
        # print('type:{}, S:{}, C:{}, E:{}, bias:{}'.format(typ, S, C, E, bias))
        # if abs(self.y - self.lstmTrace.iloc[self.index - 1, 0] * 1.5) > 1/3 * abs(self.y_pre_raw - self.y_tail_goal):
        #     reward = -100000.0
        # el
        if self.isCrashed:
            reward = -500.0
            self.isCrashed = False
        # elif interval_pre < -0.5:
        #     reward = -100
        elif not done or self.isOver:
            reward = 6 * 0.607 * S + 1 * 0.184 * C + 3 * 0.209 * E - bias
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                print("""
                You are calling 'step()' even though this environment has already returned
                done = True. You should always call 'reset()' once you receive 'done = True'
                Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.steps_beyond_done = None

        self.isCrashed = False
        self.isOver = False

        self.x = 0
        self.y = 0
        self.index = 1
        self.overIndex = 1

        while not self.obsQueue.empty():
            self.obsQueue.get()

        while not self.actQueue.empty():
            self.actQueue.get()

        self.v_pre_goal = 0.0
        self.v_tail_goal = 0.0
        self.v_pre_raw = 0.0
        self.y_pre_goal = 0.0
        self.y_tail_goal = 0.0

        self.y_pre_goal_old = 0.0
        self.y_tail_goal_old = 0.0

        self.state = self.init_env_state
        self.action = self.init_env_action

        return np.array(self.state)

