import paddle.fluid as fluid
import parl
# from parl import layers
import copy


class DDPG(parl.Algorithm):
    def __init__(self, model, gamma=None, actor_lr=None, critic_lr=None, tau=None):
        self.model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(gamma, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)
        assert isinstance(tau, float)
        self.gamma = gamma   # 待调参
        self.actor_lr = actor_lr   # 待调参
        self.critic_lr = critic_lr  # 待调参
        self.tau = tau  # 待调参

    def predict(self, obs):
        return self.model.policy(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        actor_cost = self._actor_learn(obs)
        critic_cost = self._critic_learn(obs, action, reward, next_obs, terminal)
        return actor_cost, critic_cost

    def _actor_learn(self, obs):
        action = self.model.policy(obs)
        Q = self.model.value(obs, action)
        cost = parl.layers.reduce_mean(-1.0 * Q)
        optimizer = fluid.optimizer.AdamOptimizer(self.actor_lr)
        optimizer.minimize(cost, parameter_list=self.model.get_actor_params())
        return cost

    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        next_action = self.target_model.policy(next_obs)
        next_Q = self.target_model.value(next_obs, next_action)

        terminal = parl.layers.cast(terminal, dtype='float32')
        target_Q = reward + (1.0 - terminal) * self.gamma * next_Q
        target_Q.stop_gradient = True

        Q = self.model.value(obs, action)
        cost = parl.layers.square_error_cost(Q, target_Q)
        cost = parl.layers.reduce_mean(cost)
        optimizer = fluid.optimizer.AdamOptimizer(self.critic_lr)
        optimizer.minimize(cost)
        return cost

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(
            self.target_model,
            decay=decay,
        )



