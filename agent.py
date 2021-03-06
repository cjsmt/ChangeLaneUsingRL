import paddle.fluid as fluid
import parl
# from parl import layers
import numpy as np


class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):

        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        super(Agent, self).__init__(algorithm)
        self.alg.sync_target(decay=0)

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.fluid_executor.run(self.pred_program, feed={'obs': obs}, fetch_list=[self.pred_act])[0]
        # act = np.argmax(act)
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        # act = np.expand_dims(act, -1)
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        cost = self.fluid_executor.run(self.learn_program, feed=feed, fetch_list=[self.critic_cost])[0]
        self.alg.sync_target()
        return cost

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = parl.layers.data(name='obs', shape=[self.obs_dim], dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = parl.layers.data(name='obs', shape=[self.obs_dim], dtype='float32')
            action = parl.layers.data(name='act', shape=[self.act_dim], dtype='float32')
            reward = parl.layers.data(name='reward', shape=[], dtype='float32')
            next_obs = parl.layers.data(name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = parl.layers.data(name='terminal', shape=[], dtype='bool')
            _, self.critic_cost = self.alg.learn(obs, action, reward, next_obs, terminal)
