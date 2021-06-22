import numpy as np
import pandas as pd
import argparse

from agent import Agent
from model import Model
# from algorithm import DDPG
from parl.algorithms import DDPG
from replay_memory import ReplayMemory
from env import CarChangeLaneEnv
# from app import send_ready_message, send_reset_message
import app
import time

ACTOR_LR = 1e-2  # Actor网络的 learning rate
CRITIC_LR = 1e-2  # Critic网络的 learning rate
GAMMA = 0.99  # reward 的衰减因子
TAU = 0.001  # 软更新的系数
MEMORY_SIZE = int(4e4)  # 经验池大小
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 40  # 预存一部分经验之后再开始训练
BATCH_SIZE = 50
REWARD_SCALE = 0.1  # reward 缩放系数
NOISE = 0.05  # 动作噪声方差
# TRAIN_EPISODE = 6e3  # 训练的总episode数
TRAIN_EPISODE = 1e2
env = CarChangeLaneEnv()
rpm = ReplayMemory(MEMORY_SIZE)


def init_environment(state_dic, action_dic):
    np.set_printoptions(suppress=True)
    env.init_env_state = state_dic.values.squeeze() * 1.5
    env.init_env_action = action_dic.values.squeeze() * 1.5


def setOver():
    env.isOver = True


def setFinishedReset():
    env.isReset = False


def run_step(v_pre_goal, v_tail_goal, v_pre_raw, y_pre_goal, y_tail_goal, y_pre_raw):
    if not env.obsQueue.full():
        env.obsQueue.put([v_pre_goal * 1.5, v_tail_goal * 1.5, v_pre_raw * 1.5,
                          y_pre_goal * 1.5, y_tail_goal * 1.5, y_pre_raw * 1.5])


def move_step():
    return env.actQueue.get()


# 训练一个episode
def run_episode(agent):
    print('RESET THE ENVIRONMENT {}'.format(env.epiNum))  # 通知仿真环境重启
    env.isReset = True
    obs = env.reset()
    env.epiNum += 1
    total_reward = 0
    steps = 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))

        # 增加探索扰动, 输出限制在 [-1.0, 1.0] 范围内
        action = np.clip(np.random.normal(action, NOISE), 0, 1.0)

        while True:
            if not env.obsQueue.empty():
                env.v_pre_goal, env.v_tail_goal, env.v_pre_raw, \
                env.y_pre_goal, env.y_tail_goal, env.y_pre_raw = env.obsQueue.get()
                next_obs, reward, done, info = env.step(action)
                break

        action = action.squeeze()  # 方便存入replay_memory

        rpm.append((obs, action, REWARD_SCALE * reward, next_obs, done))

        if len(rpm) > MEMORY_WARMUP_SIZE and (steps % 5) == 0:
            (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm.sample(BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)

        obs = next_obs
        total_reward += reward

        if done or steps >= len(env.lstmTrace) - 1:
            break
    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(agent):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        print('RESET THE ENVIRONMENT')  # 通知仿真环境重启
        env.isReset = True
        total_reward = 0
        steps = 0
        while True:
            steps += 1
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            action = np.clip(action, 0, 1.0)

            while True:
                if not env.obsQueue.empty():
                    env.v_pre_goal, env.v_tail_goal, env.v_pre_raw, \
                    env.y_pre_goal, env.y_tail_goal, env.y_pre_raw = env.obsQueue.get()
                    next_obs, reward, done, info = env.step(action)
                    break

            obs = next_obs
            total_reward += reward

            if done or steps >= len(env.lstmTrace) - 1:
                break
        eval_reward.append(total_reward)
        if env.isRecord:
            env.traceRecord.to_csv('trace_290_new.csv')
            env.isRecord = False
    return np.mean(eval_reward)


def start():
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env.traceRecord.to_csv('ccc.csv')

    # 使用PARL框架创建agent
    model = Model(act_dim)
    algorithm = DDPG(model=model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = Agent(algorithm=algorithm, obs_dim=obs_dim, act_dim=act_dim)

    # 创建经验池
    print('READY TO START')
    env.isReady = True
    # 往经验池中预存数据
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(agent)
    print('MEMORY WARMUP FINISHED')
    env.epiNum = 0

    episode = 0
    del env.traceRecord
    env.traceRecord = pd.DataFrame([[0, 0]], columns=['x', 'y'])
    while episode < TRAIN_EPISODE:
        for i in range(2):
            total_reward = 0.0
            for j in range(5):
                total_reward = run_episode(agent)
                episode += 1
                env.trainReward.put(total_reward)
                print('Training episode:{}  Total reward:{}'.format(episode, total_reward))
        env.isRecord = True
        eval_reward = evaluate(agent)
        env.evalReward.put(eval_reward)
        env.isRecord = False
        print('Evaluating episode:{}   Test reward:{}'.format(episode / 20, eval_reward))
