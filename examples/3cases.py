import numpy as np
from matplotlib import pyplot as plt
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

    mu = 0.1
    beta = 0.7

    fig, ax = plt.subplots(1, 3, sharey=True)
    ax[0].plot([-1, 6], [0, 0], c="k", alpha=0.75, linewidth=1)
    ax[1].plot([-1, 6], [0, 0], c="k", alpha=0.75, linewidth=1)
    ax[2].plot([-1, 6], [0, 0], c="k", alpha=0.75, linewidth=1)

    times1 = [1.5, 5]
    ax[0].plot([0, times1[0]], [mu, mu], c="r")
    lines1 = np.linspace(times1[0], times1[1], 100)
    fun1 = lambda x : mu + (0.2)*np.exp(-beta*(x - times1[0]))
    ax[0].plot([times1[0], times1[0]], [mu, mu + 0.2], c="r")
    ax[0].plot(lines1, fun1(lines1), c="r")
    ax[0].scatter([times1[0]], [0], marker="x", linewidths=1, c="k")




    times2 = [1.5, 4, 5]
    ax[1].plot([0, times2[0]], [mu, mu], c="r")
    lines2 = np.linspace(times2[0], times2[1], 100)
    fun2 = lambda x: mu - 0.2 * np.exp(-beta * (x - times2[0]))
    ax[1].plot([times2[0], times2[0]], [0, mu - 0.2], c="#1f77b4")
    ax[1].plot([times2[0], times2[0]], [mu, 0], c="r")
    ax[1].plot(lines2, fun2(lines2), c="#1f77b4")
    ax[1].plot(lines2, np.maximum(fun2(lines2), 0), c="r")
    last2 = fun2(lines2[-1])
    lines2last = np.linspace(times2[1], times2[2], 1000)
    fun2last = lambda x: mu - 0.2 * np.exp(-beta * (x - times2[0])) + 0.1 * np.exp(-beta * (x - times2[1]))
    ax[1].plot([times2[1], times2[1]], [last2, last2 + 0.1], c="r")
    ax[1].plot(lines2last, fun2last(lines2last), c="r")
    aux = times2[0] + (1 / beta) * np.log((0.2) / (mu))
    ax[1].scatter([times2[0], times2[1]], [0, 0], marker="x", linewidths=1, c="k")
    ax[1].scatter([aux], [0], marker="x", linewidths=1, c="green")



    beta = 0.3
    times2 = [0.5, 1.5, 5]
    ax[2].plot([0, times2[0]], [mu, mu], c="r")

    fun2 = lambda x: mu - 0.3 * np.exp(-beta * (x - times2[0]))
    lines2 = np.linspace(times2[0], 5, 1000)
    ax[2].plot(lines2, fun2(lines2), c="k", alpha=0.5, linestyle="--")

    lines2 = np.linspace(times2[0], times2[1], 100)
    ax[2].plot([times2[0], times2[0]], [0, mu - 0.3], c="#1f77b4")
    ax[2].plot([times2[0], times2[0]], [mu, 0], c="r")
    ax[2].plot(lines2, fun2(lines2), c="#1f77b4")
    ax[2].plot(lines2, np.maximum(fun2(lines2), 0), c="r")
    last2 = fun2(lines2[-1])
    lines2last = np.linspace(times2[1], times2[2], 1000)
    fun2last = lambda x: mu - 0.3 * np.exp(-beta * (x - times2[0])) + 0.2 * np.exp(-beta * (x - times2[1]))
    ax[2].plot([times2[1], times2[1]], [last2, 0], c="#1f77b4")
    ax[2].plot([times2[1], times2[1]], [0, last2 + 0.2], c="r")
    ax[2].plot(lines2last, fun2last(lines2last), c="#1f77b4")
    ax[2].plot(lines2last, np.maximum(fun2last(lines2last), 0), c="r")
    aux = times2[0] + (1/beta)*np.log((0.3)/(mu))
    ax[2].scatter([times2[0], times2[1]], [0, 0],  marker="x", linewidths=1, c="k")
    ax[2].scatter([aux], [0], marker="x", linewidths=1, c="brown")

    # Create a process with given parameters and maximal number of jumps.

    # fig, ax = plt.subplots(2,1, figsize=[1.5*5.3, 1.3*3.5], sharey=True)
    # ax[0].plot([-1, 3], [0, 0], c="k", alpha=0.75, linewidth=1)
    # ax[1].plot([-1, 3], [0, 0], c="k", alpha=0.75, linewidth=1)

    # Plotting function of intensity and step functions.

    # ax[0].set_xlabel("t")
    # ax[0].set_ylabel(f"$\lambda(t)$")
    #
    # ax[0].scatter([t for t,m in hawkes.timestamps[1:4]], [0,0,0], c="k", marker="x", linewidths=1)
    # ax[1].scatter([t for t, m in hawkes.timestamps[1:4]], [0, 0, 0], c="k", marker="x", linewidths=1)
    #
    # ax[0].annotate(f"$T_1$", xy=(hawkes.timestamps[1][0], 0), xytext=(hawkes.timestamps[1][0]-0.07, -0.6),
    #             annotation_clip=False)
    # #ax[0].annotate(f"$T_1^\star$", xy=(hawkes.timestamps[1][0], 0), xytext=(hawkes.timestamps[1][0] - 0.2, hawkes.intensity_jumps[0][1]-0.15),
    #             #annotation_clip=False)
    #
    # ax[0].annotate(f"$T_2$", xy=(hawkes.timestamps[2][0], 0), xytext=(hawkes.timestamps[2][0] + 0.01, -0.6),
    #             annotation_clip=False)
    # #ax[0].annotate(f"$T_2^\star$", xy=(hawkes.timestamps[2][0], 0), xytext=(2.5, -0.06),
    #             #annotation_clip=False)
    #
    # ax[0].annotate(f"$T_3$", xy=(hawkes.timestamps[3][0], 0), xytext=(hawkes.timestamps[3][0] - 0.07, -0.6),
    #             annotation_clip=False)
    # #ax[0].annotate(f"$T_3^\star$", xy=(hawkes.timestamps[3][0], 0), xytext=(6.1, -0.06),
    #             #annotation_clip=False)
    #
    # #ax[0].annotate(f"$T_4$", xy=(hawkes.timestamps[4][0], 0), xytext=(hawkes.timestamps[4][0] - 0.15, -0.1),
    #             #annotation_clip=False)
    # #ax[0].annotate(f"$T_4^\star$", xy=(hawkes.timestamps[4][0], 0), xytext=(hawkes.timestamps[4][0] - 0.15, -0.15),
    #             #annotation_clip=False)
    #
    # ax[1].annotate(f"$T_1$", xy=(hawkes.timestamps[1][0], 0), xytext=(hawkes.timestamps[1][0]-0.07, -0.6),
    #                annotation_clip=False)
    # # ax[0].annotate(f"$T_1^\star$", xy=(hawkes.timestamps[1][0], 0), xytext=(hawkes.timestamps[1][0] - 0.2, hawkes.intensity_jumps[0][1]-0.15),
    # # annotation_clip=False)
    #
    # ax[1].annotate(f"$T_2$", xy=(hawkes.timestamps[2][0], 0), xytext=(hawkes.timestamps[2][0]+0.01, -0.6),
    #                annotation_clip=False)
    # # ax[0].annotate(f"$T_2^\star$", xy=(hawkes.timestamps[2][0], 0), xytext=(2.5, -0.06),
    # # annotation_clip=False)
    #
    # ax[1].annotate(f"$T_3$", xy=(hawkes.timestamps[3][0], 0), xytext=(hawkes.timestamps[3][0]-0.07, -0.6),
    #                annotation_clip=False)
    #
    #
    # aux = [alpha[0, m-1] for t,m in hawkes.timestamps[1:4]]
    # ax[0].scatter([t for t,m in hawkes.timestamps[1:4]], np.array(hawkes.intensity_jumps[0][1:4]) - np.array(aux), linewidths=1.5, s=60,
    #              facecolors='none', edgecolors='r')
    # ax[0].scatter([t for t, m in hawkes.timestamps[1:4]], np.array(hawkes.intensity_jumps[0][1:4]),
    #               s=15, c="r", zorder=3)
    # aux = [alpha[1, m - 1] for t, m in hawkes.timestamps[1:4]]
    # ax[1].scatter([t for t, m in hawkes.timestamps[1:4]], np.array(hawkes.intensity_jumps[1][1:4]) - np.array(aux),
    #               linewidths=1.5, s=60,
    #               facecolors='none', edgecolors='r')
    # ax[1].scatter([t for t, m in hawkes.timestamps[1:4]], np.array(hawkes.intensity_jumps[1][1:4]),
    #               s=15, c="r", zorder=3)
    # # plt.scatter([0.2, 1.7, 5.75, 11.05, 15], np.array(hawkes.intensity_jumps[1:]), s=15, c="r", zorder=3)

    ax[0].set_xlim((0, times1[1]))
    ax[1].set_xlim((0, times1[1]))
    ax[2].set_xlim((0, times1[1]))

    ax[0].set_ylabel("$\lambda^1$")

    ax[0].set_xlabel("$t$")

    #plt.savefig('cooldownTimesMarkedMulti.pdf', bbox_inches='tight', format="pdf", quality=90)

    plt.show()
