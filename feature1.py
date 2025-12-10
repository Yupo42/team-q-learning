import gymnasium
import numpy as np
import matplotlib.pyplot as plt

def getQTest(Q, env):
    # Тест обученного агента (без визуализации)
    test_episodes = 100
    wins = 0
    for _ in range(test_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(Q[state, :])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated and reward > 0:  # Успех
                wins += 1
                break

    print(f"\nАгент достиг цели в {wins}/{test_episodes} тестовых эпизодах ({wins / test_episodes * 100:.1f}%).")

    # Визуализация обученного агента (отдельная среда с render_mode)
    env_render = gymnasium.make('FrozenLake-v1', is_slippery=True, render_mode="human")
    state, _ = env_render.reset()
    done = False

    print("\nЗапуск обученного агента с визуализацией:")
    while not done:
        action = np.argmax(Q[state, :])  # Используем обученную политику
        state, _, terminated, truncated, _ = env_render.step(action)
        done = terminated or truncated

    env_render.close()