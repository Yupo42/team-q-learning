import gymnasium
import numpy as np
import matplotlib.pyplot as plt


def getQTable(Q, rewards_per_episode):
    # Визуализация средней награды

    avg_rewards = [np.mean(rewards_per_episode[max(0, i-100):i+1]) for i in range(len(rewards_per_episode))]
    plt.plot(avg_rewards)
    plt.title("Средняя награда за последние 100 эпизодов")
    plt.xlabel("Эпизод")
    plt.ylabel("Средняя награда")
    plt.show()

    # Вывод Q-таблицы для начального состояния (0)
    print("\nQ-значения для начального состояния (0):")
    print(f"Влево (0): {Q[0, 0]:.4f}, Вниз (1): {Q[0, 1]:.4f}, Вправо (2): {Q[0, 2]:.4f}, Вверх (3): {Q[0, 3]:.4f}")


