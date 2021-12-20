import numpy as np
import matplotlib.pyplot as plt

def plot_avg_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('PPO_score_history_average_100')
    plt.savefig(figure_file)
def plot_learning_curve(x, scores, figure_file):
    plt.plot(x, scores)
    plt.title('PPO_score_history')
    plt.savefig(figure_file)