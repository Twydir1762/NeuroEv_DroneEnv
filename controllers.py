import numpy as np
import random

class BaseController:
    def __init__(self, eps=1.0, gamma=0.99,
                 lr=0.1, eps_decay=0.997, min_eps=0.01,
                 map_size=(24, 12), charge_pos=(21.5, 8.5),
                 n_offsets=14, b_bins=12, acts_num=12,
                 velocities=(0.10, 0.25, 0.40, 0.60)):

        # (dx, dy, b, vel, act)
        self.q_table = np.zeros((n_offsets, n_offsets, b_bins, len(velocities), acts_num))
        self.eps = eps
        self.gamma = gamma
        self.lr = lr
        self.eps_decay = eps_decay
        self.min_eps = min_eps

        self.dx_bins = np.linspace(-charge_pos[0],map_size[0] - charge_pos[0], n_offsets + 1)[1:-1]
        self.dy_bins = np.linspace(-charge_pos[1],map_size[1] - charge_pos[1], n_offsets + 1)[1:-1]
        self.b_bins = np.linspace(0, 1 , b_bins + 1)[1:-1]
        self.vels = np.array(velocities)
        self.actions = tuple(range(acts_num))

    def get_discrete_state(self, obs):
        dx, dy, b, vel = obs

        dx_bin = int(np.digitize(dx, self.dx_bins))
        dy_bin = int(np.digitize(dy, self.dy_bins))
        b_bin = int(np.digitize(b, self.b_bins))

        vel_idx = np.argmin(np.abs(self.vels - vel))

        return dx_bin, dy_bin, b_bin, vel_idx

    def get_q(self, state):
        return self.q_table[state]

    def update_q(self, state, next_state, action, reward, done=False):
        raise NotImplementedError

    def choose_action(self, state):
        if self.eps > random.random():
            return random.choice(self.actions)

        return np.argmax(self.get_q(state))

    def update_eps(self):
        if self.eps > self.min_eps: self.eps *= self.eps_decay

    def save_q(self, save_path):
        np.save(save_path, self.q_table)

    def load_q(self, path):
        self.q_table = np.load(path)

class RationalAgent(BaseController):
    def update_q(self, state, next_state, action, reward, done=False):
        q_s = self.get_q(state)
        q_s_next = self.get_q(next_state)
        q_s_max = np.max(q_s_next) if not done else 0

        self.q_table[state][action] += self.lr * ((reward + self.gamma * q_s_max) - q_s[action])

class ProspectTheoryAgent(BaseController):
    def __init__(self, alpha_p=0.88, beta_p=0.88, lambda_p=2.35, **kwargs):
        super().__init__(**kwargs)
        self.alpha_p = alpha_p
        self.beta_p = beta_p
        self.lambda_p = lambda_p

    def _prospect_val(self, reward):
        if reward >= 0:
            return reward ** self.alpha_p
        else:
            return -self.lambda_p * ((-reward) ** self.beta_p)

    def update_q(self, state, next_state, action, reward, done=False):
        v_reward = self._prospect_val(reward)
        q_s = self.get_q(state)
        q_s_next = self.get_q(next_state)
        q_s_max = np.max(q_s_next) if not done else 0

        self.q_table[state][action] += self.lr * ((v_reward + self.gamma * q_s_max) - q_s[action])

class RiskSensitiveAgent(BaseController):
    def __init__(self, eta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.eta = eta

    def update_q(self, state, next_state, action, reward, done=False):
        r_norm = reward / 100.0

        if done:
            u_next_min = 1.0
        else:
            u_next_min = np.min(np.exp(-self.eta * self.get_q(next_state)))

        u_target = np.exp(-self.eta * r_norm) * (u_next_min ** self.gamma)
        u_current = np.exp(-self.eta * self.q_table[state][action])
        u_new = (1 - self.lr) * u_current + self.lr * u_target

        # клип для логарифма
        u_new = min(max(u_new, 1e-10), 1e10)

        # Обратное преобразование
        self.q_table[state][action] = -1 / self.eta * np.log(u_new)

