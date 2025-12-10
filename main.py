import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import feature1
import feature2

# Создаём среду без визуализации для обучения
env = gymnasium.make('FrozenLake-v1', is_slippery=True)

n_states = env.observation_space.n  # 16
n_actions = env.action_space.n      # 4

# Инициализация Q-таблицы нулями
Q = np.zeros((n_states, n_actions))

# Гиперпараметры
alpha = 0.1
gamma = 0.99

# Эпсилон-затухание
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
epsilon = epsilon_start

# Для анализа наград
rewards_per_episode = []

# Обучение
episodes = 10000
for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Epsilon-greedy выбор действия
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # Случайное действие
        else:
            action = np.argmax(Q[state, :])  # Жадное действие

        # Шаг в среде
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Обновление Q-таблицы
        Q[state, action] += alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
        state = new_state

    rewards_per_episode.append(total_reward)

    # Обновление epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("Обучение завершено.")

feature2.getQTable()
feature1.getQTest()
