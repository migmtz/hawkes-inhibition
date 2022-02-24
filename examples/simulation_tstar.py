import numpy as np
from matplotlib import pyplot as plt
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
import matplotlib
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "dejavuserif"
import seaborn as sns


if __name__ == "__main__":
    # Set seed
    matplotlib.rcParams.update({'font.size': 14})
    sns.set_theme()

    np.random.seed(3)

    dim = 2

    mu = np.array([0.8, 1.0])
    alpha = np.array([[-1.9, 3], [0.9, -0.7]])
    beta = np.array([[2, 20], [3, 2]])

    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=15 * (dim - 1))

    # Create a process with given parameters and maximal number of jumps.

    hawkes.simulate()

    fig, ax = plt.subplots(1,2, figsize=[1.5*5.3, 1.2*3.5])

    # Plotting function of intensity and step functions.
    hawkes.plot_intensity(ax=ax, plot_N=False)

    ax[0].set_xlabel("t")
    ax[0].set_ylabel(f"$\lambda(t)$")

    ax[0].annotate(f"$T_1$", xy=(hawkes.timestamps[1][0], 0), xytext=(hawkes.timestamps[1][0]-0.2, hawkes.intensity_jumps[0][1]-0.1),
                annotation_clip=False)
    ax[0].annotate(f"$T_1^\star$", xy=(hawkes.timestamps[1][0], 0), xytext=(hawkes.timestamps[1][0] - 0.2, hawkes.intensity_jumps[0][1]-0.15),
                annotation_clip=False)

    ax[0].annotate(f"$T_2$", xy=(hawkes.timestamps[2][0], 0), xytext=(hawkes.timestamps[2][0] - 0.75, -0.06),
                annotation_clip=False)
    ax[0].annotate(f"$T_2^\star$", xy=(hawkes.timestamps[2][0], 0), xytext=(2.5, -0.06),
                annotation_clip=False)

    ax[0].annotate(f"$T_3$", xy=(hawkes.timestamps[3][0], 0), xytext=(hawkes.timestamps[3][0] - 0.75, -0.06),
                annotation_clip=False)
    ax[0].annotate(f"$T_3^\star$", xy=(hawkes.timestamps[3][0], 0), xytext=(6.1, -0.06),
                annotation_clip=False)

    ax[0].annotate(f"$T_4$", xy=(hawkes.timestamps[4][0], 0), xytext=(hawkes.timestamps[4][0] - 0.15, -0.06),
                annotation_clip=False)
    ax[0].annotate(f"$T_4^\star$", xy=(hawkes.timestamps[4][0], 0), xytext=(hawkes.timestamps[4][0] - 0.15, -0.15),
                annotation_clip=False)

    # plt.scatter([0.2, 1.7, 5.75, 11.05, 15], np.array(hawkes.intensity_jumps[1:]) - alpha, linewidths=1.5, s=60,
    #             facecolors='none', edgecolors='r')
    # plt.scatter([0.2, 1.7, 5.75, 11.05, 15], np.array(hawkes.intensity_jumps[1:]), s=15, c="r", zorder=3)

    ax[0].set_xlim((0,2))

    # plt.savefig('cooldownTimesMarkedSerif2.png', bbox_inches='tight', format="png", quality=90)

    plt.show()
