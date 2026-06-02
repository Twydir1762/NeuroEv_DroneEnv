from dronenv import DroneEnv
from controllers import RationalAgent, RiskSensitiveAgent, ProspectTheoryAgent
import random
import numpy as np

""" Среда """
SEED = 42

EPISODES = 25000
TEST_EPISODES = 300
MAX_STEPS = 200
CHARGE_RADIUS = 0.9 # 0.9
MIN_CHARGE = 0.04
WIN_STEPS =  140 # 140
WIN_CHARGE = 0.9 # 0.9
WIND_MAX = 0.11

""" Гиперпараметры """
# Общие
EPS = 1.0
MIN_EPS = 0.01
EPS_DECAY = 0.9997 # 0.997
GAMMA = 0.99

# Поведенческий агент
ALPHA_P = 0.95 # 0.88
BETA_P = 0.95 # 0.88
LAMBDA_P = 1.5 # 2.35

# Риск-чувствительный агент
ETA = 0.5 # 1.0

""" Тест """
CON_NAME = 'pt_agent'
FPS = 20

random.seed(SEED)
np.random.seed(SEED)

cons = {
    'r_agent': RationalAgent(eps=EPS, min_eps=MIN_EPS, eps_decay=EPS_DECAY, gamma=GAMMA),
    'pt_agent': ProspectTheoryAgent(eps=EPS, min_eps=MIN_EPS,
                                    eps_decay=EPS_DECAY, gamma=GAMMA,
                                    alpha_p=ALPHA_P, beta_p=BETA_P, lambda_p=LAMBDA_P),
    'rs_agent': RiskSensitiveAgent(eps=EPS, min_eps=MIN_EPS, eps_decay=EPS_DECAY, gamma=GAMMA, eta=ETA)
}

env = DroneEnv(render_mode='human', fps=FPS, size_mult=50)
env.charge_radius = CHARGE_RADIUS
env.max_steps = MAX_STEPS
env.min_charge = MIN_CHARGE
env.win_steps = WIN_STEPS
env.win_charge = WIN_CHARGE
env.wind_max = WIND_MAX
env.reset_graphics()

dcon = cons[CON_NAME]
dcon.load_q(f"{CON_NAME}.npy")
dcon.eps = 0.005

wins = 0

for _ in range(100):
    done = False
    obs = env.reset()

    while not done:
        state = dcon.get_discrete_state(obs)
        action = dcon.choose_action(state)

        obs, reward, terminated, info = env.step(action)

        if terminated:
            done = True
            if reward > 0:
                wins += 1

        env.render()

print(wins)