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

    #np.random.seed(3)
    np.random.seed(4)

    dim = 2

    # mu = np.array([0.8, 1.0])
    # alpha = np.array([[-1.9, 3], [0.9, -0.7]])
    # beta = np.array([[2, 20], [3, 2]])
    mu = np.array([1.2, 0.4])
    alpha = np.array([[-3.0, 2.0], [1.2, -2.7]])
    beta = np.array([[2.5, 2.5], [1.6, 1.6]])

    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=15 * (dim - 1))

    # Create a process with given parameters and maximal number of jumps.

    hawkes.simulate()
    print(hawkes.timestamps[0:4])

    fig, ax = plt.subplots(2,1, figsize=[12, 8], sharey=True)
    ax[0].plot([-1, 10], [0, 0], c="k", alpha=0.75, linewidth=1)
    ax[1].plot([-1, 10], [0, 0], c="k", alpha=0.75, linewidth=1)

    # Plotting function of intensity and step functions.
    hawkes.plot_intensity(ax=ax, plot_N=False)

    list_times = [t for t,m in hawkes.timestamps[:-1] if t > 8.0]
    list_intensities_1 = [i for (t, m), i in zip(hawkes.timestamps[:-1], hawkes.intensity_jumps[0]) if t > 8.0]
    list_intensities_2 = [i for (t, m), i in zip(hawkes.timestamps[:-1], hawkes.intensity_jumps[1]) if t > 8.0]

    aux = [alpha[0, m-1] for t,m in hawkes.timestamps[1:4]]

    print(len(list_times), len(list_intensities_1), len(list_intensities_2))

    #ax[0].scatter(list_times, list_intensities_1)
    aux1 = [list_times[0] + (1/beta[0,0])*np.log((mu[0] - list_intensities_1[0])/mu[0]),
            list_times[2],
            list_times[3] + (1/beta[0,0])*np.log((mu[0] - list_intensities_1[3])/mu[0])]
    ajuste = 0.017
    altura = - 0.6
    ax[0].annotate("$T_{(1)}^{1\star}$", xy=(aux1[0]+ajuste, 0), xytext=(aux1[0]-0.05+ajuste, altura),
                   annotation_clip=False)
    ax[0].annotate("$T_{(2)}^{1\star}$", xy=(aux1[1]-0.1+ajuste, 0), xytext=(aux1[1]-0.1+ajuste, altura),
                   annotation_clip=False)
    ax[0].annotate("$T_{(3)}^{1\star}$", xy=(aux1[1]+ajuste, 0), xytext=(aux1[1]+ajuste, altura),
                   annotation_clip=False)
    ax[0].annotate("$T_{(4)}^{1\star}$", xy=(aux1[2]-0.07+ajuste, 0), xytext=(aux1[2]-0.07+ajuste, altura),
                   annotation_clip=False)

    ax[0].scatter(aux1, [0 for i in aux1], marker="x", c="k", linewidths=1, label="Restart times")

    aux1 = [list_times[0],
            list_times[1], list_times[3]]
    ax[1].annotate("$T_{(1)}^{2\star}$", xy=(aux1[0]-0.07+ajuste, 0), xytext=(aux1[0]-0.07+ajuste, altura),
                   annotation_clip=False)
    ax[1].annotate("$T_{(2)}^{2\star}$", xy=(aux1[1] - 0.07+ajuste, 0), xytext=(aux1[1] - 0.07+ajuste, altura),
                   annotation_clip=False)
    ax[1].annotate("$T_{(3)}^{2\star}$", xy=(aux1[2]-0.1+ajuste, 0), xytext=(aux1[2]-0.1+ajuste, altura),
                   annotation_clip=False)
    ax[1].annotate("$T_{(4)}^{2\star}$", xy=(aux1[2]+ajuste, 0), xytext=(aux1[2]+ajuste, altura),
                   annotation_clip=False)
    ax[1].scatter(aux1, [0 for i in aux1], marker="x", c="k", linewidths=1)

    ax[0].set_ylabel("$\lambda^1$")
    ax[1].set_ylabel("$\lambda^2$")

    ax[1].set_xlabel("$t$")

    #ax[0].set_xticks([])
    ax[0].set_xticks(list_times)
    ax[0].set_xticklabels([])
    ax[1].set_xticks(list_times)
    ax[1].set_xticklabels(["$T_{("+str(k+1) + ")}$" for k,_ in enumerate(list_times)])

    ax[0].set_xlim((8, 10))
    ax[1].set_xlim((8, 10))


    plt.savefig('restarTimesMarkedMulti.pdf', bbox_inches='tight', format="pdf", quality=90)
    ax[0].legend()
    plt.show()
