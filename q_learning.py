from dronenv import DroneEnv
from controllers import RationalAgent, ProspectTheoryAgent, RiskSensitiveAgent
from tqdm import tqdm
import sys
import mlflow
from utils import plot_learning_curve, plot_joint_cdf, plot_charge_heatmap, plot_paths, boxplot
import scipy.stats as stats
import itertools
import random
import numpy as np


""" Среда """
SEED = 42

EPISODES = 25000
TEST_EPISODES = 300
MAX_STEPS = 200
CHARGE_RADIUS = 0.9 # 2.7
MIN_CHARGE = 0.04
WIN_STEPS =  140
WIN_CHARGE = 0.9 # 0.75
WIND_MAX = 0.11

""" MLFlow """
LOG_FREQUENCY = 50
RUN_SUFFIX = "_12acts_shaping_reward_2"

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
ETA = 1.0 # 0.5

# Reward Shaping
SHAPING_COEF = 1.2


random.seed(SEED)
np.random.seed(SEED)

def train(env, dcon, episodes, q_table_path):
    pbar = tqdm(range(1, episodes+1), desc="Training", file=sys.stdout)

    reward_history = [] # Для learning curve

    charge_pos = np.array(env.charge_pos) # SHAPING

    for episode in pbar:
        # === CURRICULUM ===
        # prev_rad = env.charge_radius
        # if episode < 5000:
        #     env.charge_radius = 2.7
        # elif episode < 10000:
        #     env.charge_radius = 2.2
        # elif episode < 15000:
        #     env.charge_radius = 1.7
        # elif episode < 20000:
        #     env.charge_radius = 1.2
        # else:
        #     env.charge_radius = 0.9
        #
        # if env.charge_radius < prev_rad:
        #     dcon.eps = 0.5
        # ===================

        prev_pos = None # SHAPING

        done = False
        obs = env.reset()

        ep_reward = 0
        ep_steps = 0
        ep_collisions = 0
        final_charge = 0.0

        while not done:
            state = dcon.get_discrete_state(obs)
            action = dcon.choose_action(state)

            obs, reward, terminated, info = env.step(action)
            next_state = dcon.get_discrete_state(obs)

            # ===== SHAPING =====
            current_pos = np.array(info["drone_pos"])

            if prev_pos is None:
                shaping_bonus = 0.0
            else:
                dist_old = np.linalg.norm(prev_pos - charge_pos)
                dist_new = np.linalg.norm(current_pos - charge_pos)
                shaping_bonus = (dist_old - dist_new) * SHAPING_COEF # перебивание штрафа за шаг

            prev_pos = current_pos
            shaped_reward = reward + shaping_bonus
            # ===================

            ep_steps += 1
            ep_reward += reward

            if info["collision"]:
                ep_collisions += 1

            if terminated:
                done = True
                final_charge = info["charge"]

            # dcon.update_q(state, next_state, action, reward, done)
            dcon.update_q(state, next_state, action, shaped_reward, done) # SHAPING

        reward_history.append(ep_reward)  # Для learning curve

        if episode % LOG_FREQUENCY == 0:
            mlflow.log_metrics({
                "Reward": ep_reward,
                "Steps": ep_steps,
                "Collisions": ep_collisions,
                "Final_Charge": final_charge #, "Charge radius": env.charge_radius # CURRICULUM
            }, step=episode)

        dcon.update_eps()
        pbar.set_postfix(eps=f"{dcon.eps:.3f}", sum_reward=f"{ep_reward:.3f}",
                         collisions=f"{ep_collisions:.3f}", charge=f"{final_charge:.3f}")

    dcon.save_q(q_table_path)

    return reward_history

def test(env, dcon, episodes, q_table_path):
    dcon.load_q(q_table_path)

    win_steps_list = [] # CDF
    final_charges = [] # Heatmap
    final_positions = []
    discharges = [] # t-test
    paths = []

    wins = 0
    win_steps = 0
    total_discharge = 0.0
    total_collisions = 0

    for _ in range(episodes):
        done = False
        obs = env.reset()

        ep_collisions = False # коллизии
        ep_path = [] # траектории
        steps = 0

        while not done:
            state = dcon.get_discrete_state(obs)
            action = dcon.choose_action(state)

            obs, reward, terminated, info = env.step(action)
            steps += 1

            ep_path.append(info["drone_pos"]) # траектории

            if info["collision"]:
                ep_collisions = True

            if terminated:
                final_charge = info["charge"]
                final_charges.append(final_charge)
                final_positions.append(info["drone_pos"])

                ep_discharge = (1.0 - final_charge)
                total_discharge += ep_discharge
                discharges.append(ep_discharge)

                if reward == env.success_reward:
                    wins += 1
                    win_steps += steps
                    win_steps_list.append(steps)

                done = True

        if ep_collisions:
            total_collisions += 1

        # Траектории
        if len(paths) < 4:
            paths.append(tuple(ep_path))

    winrate = (wins / episodes) * 100
    avg_win_steps = (win_steps / wins) if wins > 0 else 0
    avg_energy = total_discharge / episodes
    collisions_rate = (total_collisions/episodes) * 100

    results = {
        "winrate": winrate,
        "avg_win_steps": avg_win_steps,
        "avg_energy": avg_energy,
        "collisions_rate": collisions_rate,
        "win_steps": win_steps_list,
        "final_charges": final_charges,
        "final_positions": final_positions,
        "discharges": discharges,
        "paths": tuple(paths)
    }

    return results

cons = {
    'r_agent': RationalAgent(eps=EPS, min_eps=MIN_EPS, eps_decay=EPS_DECAY, gamma=GAMMA),
    'pt_agent': ProspectTheoryAgent(eps=EPS, min_eps=MIN_EPS,
                                    eps_decay=EPS_DECAY, gamma=GAMMA,
                                    alpha_p=ALPHA_P, beta_p=BETA_P, lambda_p=LAMBDA_P),
    'rs_agent': RiskSensitiveAgent(eps=EPS, min_eps=MIN_EPS, eps_decay=EPS_DECAY, gamma=GAMMA, eta=ETA)
}

# Mlflow сервер
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(experiment_name="Drone_QLearning_Experiment_report")

""" === Эксперимент === """
train_rewards_dict = {}
win_steps_dict = {}
final_charges_dict = {}
final_positions_dict = {}
discharges_dict = {}
paths_dict = {}

# Из среды
env_map_size = None
env_obstacles = None
env_charge_pos = None
env_charge_rad = None
data_collect = False

for con_name, con in cons.items():
    random.seed(SEED)
    np.random.seed(SEED)

    denv = DroneEnv(render_mode=None)
    denv.charge_radius = CHARGE_RADIUS
    denv.max_steps = MAX_STEPS
    denv.min_charge = MIN_CHARGE
    denv.win_steps = WIN_STEPS
    denv.win_charge = WIN_CHARGE
    denv.wind_max = WIND_MAX

    if not data_collect:
        env_map_size = denv.map_size
        env_obstacles = denv.obstacles
        env_charge_pos = denv.charge_pos
        env_charge_rad = denv.charge_radius
        data_collect = True

    # MLFlow
    with mlflow.start_run(run_name=con_name + RUN_SUFFIX):
        # Инициализация параметров для логирования
        params = {
            "seed": SEED,
            "shaping_coff": SHAPING_COEF, # SHAPING
            "gamma": con.gamma,
            "min_eps": con.min_eps,
            "eps_decay": con.eps_decay,
            "train_episodes": EPISODES,
            "test_episodes": TEST_EPISODES
        }
        if hasattr(con, "alpha_p"):
            params.update({"alpha_p": con.alpha_p, "beta_p": con.beta_p, "lambda_p": con.lambda_p})
        if hasattr(con, "eta"):
            params.update({"eta": con.eta})

        mlflow.log_params(params)

        # Обучение
        print(f"\n=== Обучение: {con_name} ===")
        train_rewards = train(denv, con, EPISODES, con_name)
        mlflow.log_artifact(f"{con_name}.npy")

        # Тестирование
        print(f"=== Тестирование: {con_name} ===")
        con.eps = 0.005 # ТЗ

        test_res = test(denv, con, TEST_EPISODES, f'{con_name}.npy')

        # Learning Curve
        train_rewards_dict[con_name] = train_rewards

        # CDF график
        win_steps_dict[con_name] = test_res["win_steps"]

        # HeatMap
        final_charges_dict[con_name] = test_res["final_charges"]
        final_positions_dict[con_name] = test_res["final_positions"]

        # Коэффициент Стьюдента
        discharges_dict[con_name] = test_res["discharges"]

        # Траектории
        paths_dict[con_name] = test_res["paths"]

        print(f"Winrate: {test_res["winrate"]:.1f}%, "
              f"Шагов до победы: {test_res["avg_win_steps"]:.1f}, "
              f"Коллизии: {test_res["collisions_rate"]:.1f}%")

        mlflow.log_metrics({
            "Winrate": test_res["winrate"],
            "Avg_Steps_to_Win": test_res["avg_win_steps"],
            "Avg_Energy_Consumed": test_res["avg_energy"],
            "Collision_Rate": test_res["collisions_rate"]
        })

with mlflow.start_run(run_name="joint_plots" + RUN_SUFFIX):
    # Learning Curve
    lc_fig = plot_learning_curve(train_rewards_dict, window=150)
    lc_fig.write_html("learning_curve.html")
    mlflow.log_artifact("learning_curve.html")

    # CDF
    cdf_fig = plot_joint_cdf(win_steps_dict)
    cdf_fig.write_html("cdf.html")
    mlflow.log_artifact("cdf.html")

    # Heatmap
    hm_fig = plot_charge_heatmap(final_positions_dict=final_positions_dict,
                                 final_charges_dict=final_charges_dict,
                                 map_size=env_map_size,
                                 charge_pos=env_charge_pos,
                                 charge_radius=env_charge_rad,
                                 obstacles=env_obstacles,
                                 subplot_width=500, max_cols=3)

    hm_fig.write_html("heatmap_charge.html")
    mlflow.log_artifact("heatmap_charge.html")

    # Траектории
    paths_fig = plot_paths(paths_dict, obstacles=env_obstacles,
                           charge_radius=env_charge_rad, charge_pos=env_charge_pos)
    paths_fig.write_html("drone_paths.html")
    mlflow.log_artifact("drone_paths.html")

    # T-test
    agents = list(discharges_dict.keys())
    report = "Результаты попарного t-test (расход энергии):\n\n"

    for agent1, agent2 in itertools.combinations(agents, 2):
        data1 = discharges_dict[agent1]
        data2 = discharges_dict[agent2]

        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)

        report += (f"{agent1}_vs_{agent2}:\nt-stat: {t_stat:.4f}\n"
                   f"p-value: {p_value:.4f}\nРазница: {"ЗНАЧИМО" if p_value < 0.05 else "НЕ ЗНАЧИМО"}\n\n")

    with open("t_test_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    mlflow.log_artifact("t_test_report.txt")
    print(report)

    box_fig = boxplot(discharges_dict)
    box_fig.write_html("boxplot.html")
    mlflow.log_artifact("boxplot.html")
