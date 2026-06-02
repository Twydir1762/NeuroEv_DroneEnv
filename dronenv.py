import random
import math
import json

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import sys
import logging

logger = logging.getLogger(__name__)

def prepare_map(map_path):
    try:
        with open(map_path, 'r') as f:
            config_data = json.load(f)

        res = {
            'map_size': tuple(config_data['map_size']),
            'start_pos': tuple(config_data['start_pos']),
            'charge_pos': tuple(config_data['charge_pos'])
        }
        obstacles = []
        for obs in config_data['obstacles']:
            obstacles.append(tuple(obs))

        res['obstacles'] = tuple(obstacles)
        return res

    except FileNotFoundError:
        logger.error(f"Map file not found: {map_path}")
        raise
    except (json.decoder.JSONDecodeError, TypeError, KeyError) as e:
        logger.error(f"Invalid map config file: {map_path} - {e}")
        raise

class Drone:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0.0
        self.vel = 0.0
        self.charge = 1.0

class DroneEnv:
    def __init__(self,
                 charge_radius=0.9,
                 velocities=(0.10, 0.25, 0.40, 0.60),
                 angles=(0.0, math.pi / 2.0, math.pi, 3 * math.pi / 2.0),
                 render_mode=None, size_mult=10, fps=30, map_path='map.json'):
        map_cfg = prepare_map(map_path)

        # Система
        self.max_steps = 200
        self.min_charge = 0.04
        self.win_steps = 140
        self.win_charge = 0.90

        # Награды
        self.success_reward = 90
        self.discharge_reward = -170

        # Ветер
        self.wind_change_freq = 15
        self.wind_max = 0.11

        """ Переменные среды """
        self.steps = 0
        self.wind_x = 0.0
        self.wind_y = 0.0
        self.velocities = velocities
        self.drone = None
        self._angles = angles
        self.action_space = self._build_action_space()

        """ Карта """
        self.map_size = map_cfg['map_size']

        # Зарядка
        self.charge_pos = map_cfg['charge_pos']
        self.charge_radius = charge_radius
        self.charge_power = 0.025

        # Препятствия
        # x1, y1, x2, y2
        self.obstacles = map_cfg['obstacles']

        # Точка спавна
        self.start_pos = map_cfg['start_pos']
        self.disp = 1.8
        self.clip_x = (1.0, 7.0)
        self.clip_y = (2.0, 10.0)

        # Стартовый ресет
        self.reset()

        """ Визуализация """
        self.render_mode = render_mode
        self.size_mult = size_mult
        self.fps = fps
        self._screen = None
        self._clock = None
        self._map_surf = None
        self._drone_surf = None
        self._aura_surf = None
        self.colors = {
            'back': (25, 28, 40),
            'drone': (255, 160, 0),
            'drone_front': (40, 40, 60),
            'obstacles': (60, 65, 80),
            'charge_station': (80, 120, 255),
            'charge_aura': (100, 150, 255),
            'charge_ind_back': (220, 60, 60),
            'charge_ind': (60, 220, 60)
        }

        # Индикатор зарядки дрона (координаты)
        self._cv_x = int(self.map_size[0] * self.size_mult * 0.9)
        self._cv_y = int(self.map_size[1] * self.size_mult * 0.02)
        self._cv_max_w = int(self.map_size[0] * self.size_mult * 0.09)
        self._cv_h = int(self.map_size[1] * self.size_mult * 0.02)

        if self.render_mode == 'human':
            pygame.init()
            self.reset_graphics()

    def _build_action_space(self):
        actions_list = []
        # Действия 0-11 (4 направления * 3 скорости)
        for angle in self._angles:
            for v in self.velocities[:-1]:
                actions_list.append((angle, v))

        # "Служебные комбинации" - в разработке, я не понял что это
        # for angle in self._angles:
        #     actions_list.append((angle, 0))

        return tuple(actions_list)

    def _update_wind(self):
        self.wind_x = random.uniform(-self.wind_max, self.wind_max)
        self.wind_y = random.uniform(-self.wind_max, self.wind_max)

    def get_obs(self):
        dx = self.drone.x - self.charge_pos[0]
        dy = self.drone.y - self.charge_pos[1]

        return dx, dy, self.drone.charge, self.drone.vel

    def reset(self):
        x = random.gauss(self.start_pos[0], self.disp)
        y = random.gauss(self.start_pos[1], self.disp)

        x = min(self.clip_x[1], max(self.clip_x[0], x))
        y = min(self.clip_y[1], max(self.clip_y[0], y))

        self.steps = 0
        self.drone = Drone(x, y)
        self.wind_x = 0.0 # ветер сброс
        self.wind_y = 0.0

        return self.get_obs()

    def apply_action(self, action):
        angle, vel = self.action_space[action]
        self.drone.angle = angle
        self.drone.vel = vel

    def _check_obstacle(self, x, y):
        for (x1, y1, x2, y2) in self.obstacles:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    def _check_charge_zone(self, x, y):
        dx = x - self.charge_pos[0]
        dy = y - self.charge_pos[1]
        return math.sqrt(dx**2 + dy**2) < self.charge_radius

    def step(self, action):
        self.steps += 1
        done = False
        collision = False

        # Действие агента
        self.apply_action(action)

        # Перемена ветра
        if self.steps % self.wind_change_freq == 0:
            self._update_wind()

        new_x = self.drone.x + self.drone.vel * math.cos(self.drone.angle) + self.wind_x
        new_y = self.drone.y + self.drone.vel * math.sin(self.drone.angle) + self.wind_y

        # Столкновения
        if self._check_obstacle(new_x, new_y):
            collision = True
            new_x = self.drone.x
            new_y = self.drone.y

        # Клип координат чтобы не сдуло за карту
        self.drone.x = min(self.map_size[0], max(0.0, new_x))
        self.drone.y = min(self.map_size[1], max(0.0, new_y))

        # Зарядка
        in_station = self._check_charge_zone(self.drone.x, self.drone.y)

        self.drone.charge = min(1, self.drone.charge
                                - 0.018 * self.drone.vel**2
                                - 0.003 * (self.wind_x**2 + self.wind_y**2) # норма вектора в квадрате
                                - 0.002 * abs(self.drone.vel)
                                + (self.charge_power if in_station else 0))

        # Проверка терминальных состояний
        if in_station and self.drone.charge > self.win_charge and self.steps <= self.win_steps:
            reward = self.success_reward
            done = True
        elif self.drone.charge < self.min_charge or self.steps >= self.max_steps:
            reward = self.discharge_reward
            done = True
        else:
            reward = -1.0 - 0.03 * self.drone.vel**2 + 0.12 * (self.charge_power if in_station else 0)

        obs = self.get_obs()
        info = {
            'collision': collision,
            'in_station': in_station,
            'steps': self.steps,
            'charge': self.drone.charge,
            'drone_pos': (self.drone.x, self.drone.y),
        }

        return obs, reward, done, info

    #  Визуал
    def _get_w(self, x):
        # Позиция в px с учетом масштаба
        return int(x * self.size_mult)

    def _get_h(self, y):
        # Инверсия (!!!)
        return int((self.map_size[1] - y) * self.size_mult)

    def reset_graphics(self):
        # Системное
        screen_size = (int(self.map_size[0] * self.size_mult), int(self.map_size[1] * self.size_mult))
        self._screen = pygame.display.set_mode(screen_size)
        self._clock = pygame.time.Clock()

        # Карта
        self._map_surf = pygame.Surface(screen_size)
        self._map_surf.fill(self.colors['back'])

        # Препятствия
        for (x1, y1, x2, y2) in self.obstacles:
            obs_x = self._get_w(x1)
            obs_y = self._get_h(y2)
            obs_w = self._get_w(x2 - x1)
            obs_h = self._get_h(y1) - obs_y
            pygame.draw.rect(self._map_surf,
                self.colors['obstacles'],
                (obs_x, obs_y, obs_w, obs_h))

        # Подзарядка
        c_r = int(0.15 * self.size_mult)
        pygame.draw.circle(
            self._map_surf,
            self.colors['charge_station'],
            (self._get_w(self.charge_pos[0]), self._get_h(self.charge_pos[1])),
             c_r
        )

        # Аура подарядки
        a_r = int(self.charge_radius * self.size_mult)
        self._aura_surf = pygame.Surface((a_r * 2, a_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(self._aura_surf,
                           (*self.colors['charge_aura'], 75),
                           (a_r, a_r), a_r)

        # Фон индикатора заряда
        pygame.draw.rect(self._map_surf,
                         self.colors['charge_ind_back'],
                         (self._cv_x, self._cv_y,
                          self._cv_max_w, self._cv_h))

        # Дрон
        drone_r = int(0.1 * self.size_mult)
        self._drone_surf = pygame.Surface((drone_r * 2, drone_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            self._drone_surf,
            self.colors['drone'],
            (drone_r, drone_r),
            drone_r
        )
        pygame.draw.circle(
            self._drone_surf,
            self.colors['drone_front'],
            (drone_r + drone_r//1.5, drone_r), drone_r // 2,
        )

    def render(self):
        if not self._screen: return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self._clock.tick(self.fps)

        # Карта
        self._screen.blit(self._map_surf, (0, 0))

        # Индикатор ветра
        wv_r = (self.map_size[0] * self.size_mult * 0.1) // 2
        pygame.draw.circle(
            self._screen,
            (255, 255, 255),
            (wv_r, wv_r),
            wv_r,
            1
        )
        if self.wind_x != 0 or self.wind_y != 0:
            wv_len = math.sqrt(self.wind_x**2 + self.wind_y**2)
            wv_scale = wv_len / (1.4142 * self.wind_max)
            wv_dx = (self.wind_x / wv_len) * wv_scale * wv_r
            wv_dy = (self.wind_y / wv_len) * wv_scale * wv_r

            pygame.draw.line(
                self._screen,
                (255, 255, 255),
                (wv_r, wv_r),
                (wv_r + wv_dx, wv_r - wv_dy),
                1
            )

        # Индикатор заряда дрона
        charge_val = int(self._cv_max_w * self.drone.charge) # 0.0 - 0.09
        pygame.draw.rect(self._screen,
                         self.colors['charge_ind'],
                         (self._cv_x, self._cv_y, charge_val, self._cv_h))

        # Аура
        a_r = self._aura_surf.get_width() // 2
        self._screen.blit(
            self._aura_surf,
            (self._get_w(self.charge_pos[0]) - a_r,
             self._get_h(self.charge_pos[1]) - a_r)
        )

        # Дрон
        drone_rotated = pygame.transform.rotate(self._drone_surf, math.degrees(self.drone.angle))
        drone_rect = drone_rotated.get_rect(center=(self._get_w(self.drone.x), self._get_h(self.drone.y)))
        self._screen.blit(drone_rotated, drone_rect)

        pygame.display.flip()
