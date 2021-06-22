import parl
# from parl import layers


class Model(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()


class ActorModel(parl.Model):
    def __init__(self, act_dim):    # 网络结构待调参
        hid_size = 300
        self.fc1 = parl.layers.fc(size=hid_size, act='relu')
        self.fc2 = parl.layers.fc(size=2*hid_size, act='relu')
        self.fc3 = parl.layers.fc(size=act_dim, act='sigmoid')

    def policy(self, obs):
        hid1 = self.fc1(obs)
        hid2 = self.fc2(hid1)
        means = self.fc3(hid2)
        return means


class CriticModel(parl.Model):      # 网络结构待调参
    def __init__(self):
        super().__init__()
        hid_size = 300
        self.fc1 = parl.layers.fc(size=hid_size, act='relu')
        self.fc2 = parl.layers.fc(size=2*hid_size, act=None)
        self.fc3 = parl.layers.fc(size=2*hid_size, act='relu')
        self.fc4 = parl.layers.fc(size=1, act=None)

    def value(self, obs, act):

        hid1 = self.fc1(obs)
        concat = parl.layers.concat([hid1, act], axis=1)
        hid2 = self.fc2(concat)
        hid3 = self.fc3(hid2)
        Q = self.fc4(hid3)
        Q = parl.layers.squeeze(Q, axes=[1])
        return Q
