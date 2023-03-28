from class_and_func.hawkes_process import exp_thinning_hawkes
from class_and_func.estimator_class import loglikelihood_estimator_bfgs
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    np.random.seed(2)
    mu = 1
    alpha = 0.5
    beta = 2

    fig, ax = plt.subplots(2, 1)
    hp = exp_thinning_hawkes(mu, alpha, beta, max_time=50)
    hp.simulate()
    hp.plot_intensity(ax=ax)

    real_estimation = loglikelihood_estimator_bfgs(initial_guess=np.ones(3))
    real_estimation.fit(hp.timestamps)

    steps = 40
    repet = 50
    estimated_list = np.zeros((steps, 3, repet))

    for i, noise in enumerate(np.linspace(0.05, 5, steps)):
        np.random.seed(0)
        for j in range(repet):
            ppp = exp_thinning_hawkes(noise, 0, beta, max_time= 50)
            ppp.simulate()
            #ppp.plot_intensity(ax=ax)

            parasited_times = [0.0] + np.sort(hp.timestamps[1:] + ppp.timestamps[1:]).tolist()

            paras_estimation = loglikelihood_estimator_bfgs(initial_guess=np.ones(3))
            paras_estimation.fit(parasited_times)

            estimated_list[i, :, j] += np.array(paras_estimation.res.x)

    aux = np.swapaxes(np.array([np.ones((steps,repet))*mu, np.ones((steps,repet))*alpha, np.ones((steps,repet))*beta]), 0, 1)
    estimated_list = np.abs((estimated_list - aux)/aux)
    print("")
    fig_est, ax_est = plt.subplots(3, 1, sharex=True)

    smooth = 7

    # ax_est[0].plot([0.05, 2], [mu, mu])
    # ax_est[0].plot([0.05, 2], [real_estimation.res.x[0], real_estimation.res.x[0]])
    # sns.violinplot(data=estimated_list[:, 0, :].T, ax=ax_est[0])
    ax_est[0].scatter(range(0, steps), np.mean(estimated_list[:, 0, :], axis=1))
    aux = np.convolve(np.mean(estimated_list[:, 0, :], axis=1), (1 / smooth) * np.ones(smooth))
    aux2 = [np.mean(np.mean(estimated_list[:, 0, :], axis=1)[:i + 1]) for i in range(0, smooth)]
    ax_est[0].plot(range(smooth), aux2, c="r", alpha=0.8)
    ax_est[0].plot(range(smooth - 1, len(aux) - smooth), aux[smooth - 1:-smooth], c="r", alpha=0.8)
    ax_est[0].set_xticklabels([])

    # ax_est[1].plot([0.05, 2], [0.5, 0.5])
    # ax_est[1].plot([0.05, 2], [real_estimation.res.x[1], real_estimation.res.x[1]])
    # sns.violinplot(data=estimated_list[:, 1, :].T, ax=ax_est[1])
    ax_est[1].scatter(range(0, steps), np.mean(estimated_list[:, 1, :], axis=1))
    aux = np.convolve(np.mean(estimated_list[:, 1, :], axis=1), (1 / smooth) * np.ones(smooth))
    aux2 = [np.mean(np.mean(estimated_list[:, 1, :], axis=1)[:i + 1]) for i in range(0, smooth)]
    ax_est[1].plot(range(smooth), aux2, c="r", alpha=0.8)
    ax_est[1].plot(range(smooth - 1, len(aux) - smooth), aux[smooth - 1:-smooth], c="r", alpha=0.8)
    ax_est[1].set_xticklabels([])

    # ax_est[2].plot([0.05, 2], [beta, beta])
    # ax_est[2].plot([0.05, 2], [real_estimation.res.x[2], real_estimation.res.x[2]])
    # sns.violinplot(data=estimated_list[:, 2, :].T, ax=ax_est[2])
    ax_est[2].scatter(range(0, steps), np.mean(estimated_list[:, 2, :], axis=1))
    aux = np.convolve(np.mean(estimated_list[:, 2, :], axis=1), (1 / smooth) * np.ones(smooth))
    aux2 = [np.mean(np.mean(estimated_list[:, 2, :], axis=1)[:i + 1]) for i in range(0, smooth)]
    ax_est[2].plot(range(smooth), aux2, c="r", alpha=0.8)
    ax_est[2].plot(range(smooth - 1, len(aux) - smooth), aux[smooth - 1:-smooth], c="r", alpha=0.8)

    ax_est[2].set_xticks(np.arange(0, steps, 2))
    ax_est[2].set_xticklabels([np.round(0.1*k+0.05, 2) for k in range(20)])
    plt.show()