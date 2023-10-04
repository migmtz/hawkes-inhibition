from class_and_func.hawkes_process import exp_thinning_hawkes
from class_and_func.estimator_class import loglikelihood_estimator_bfgs
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import kstest


if __name__ == "__main__":
    sns.set_theme()

    np.random.seed(2)
    mu = 1
    alpha = 0.5
    beta = 2

    noise = 0.2
    repet = 100
    max_time = 2000.0

    hp = exp_thinning_hawkes(mu, alpha, beta, max_time=max_time)
    hp.simulate()
    #hp.plot_intensity(ax=ax)

    real_estimation = loglikelihood_estimator_bfgs(initial_guess=np.ones(3))
    real_estimation.fit(hp.timestamps)

    estimated_list = np.zeros((3, repet))
    kslist = []
    avg_jumps = 0

    ksval = 0
    for j in range(repet):
        np.random.seed(j)
        ppp = exp_thinning_hawkes(noise, 0, beta, max_time=max_time)
        ppp.simulate()
        #ppp.plot_intensity(ax=ax)

        parasited_times = [0.0] + np.sort(hp.timestamps[1:] + ppp.timestamps[1:]).tolist()
        avg_jumps += len(parasited_times[1:-1])

        paras_estimation = loglikelihood_estimator_bfgs(initial_guess=np.ones(3))
        paras_estimation.fit(parasited_times)

        estimated_list[:, j] += np.array(paras_estimation.res.x)

        model = exp_thinning_hawkes(paras_estimation.res.x[0], paras_estimation.res.x[1], paras_estimation.res.x[2])
        model.set_time_intensity(hp.timestamps)
        model.compensator_transform()
        ksval += kstest(model.intervals_transformed, cdf="expon").pvalue
        #print(ksval, end=" | ")

    print("")
    kslist += [ksval/repet]
    avg_jumps /= repet

    mu_estim, alpha_estim, beta_estim = np.mean(estimated_list, axis=1)

    lambda_N = mu/(1-alpha/beta)
    lambda_estim =  mu_estim/(1-alpha_estim/beta_estim)

    print("Means :")
    print("     Real mean: ", lambda_N)
    print("     Estimated mean: ", lambda_estim)
    print("     Expected mean: ", lambda_N + noise)
    print("     Average nb of points/max_time: ", avg_jumps/max_time)

    print("(Co)Variance density :") # In reality, we are checking the second order moment
    print("     Real variance: ", lambda_N * ((alpha * (2*beta - alpha))/(2 * (beta - alpha))), beta - alpha)
    print("     Estimated variance: ", lambda_estim * ((alpha_estim * (2*beta_estim - alpha_estim))/(2 * (beta_estim - alpha_estim))), beta_estim - alpha_estim)
    print("     Expected variance: ", )
    print("     Average nb of points/max_time: ", )

    fig, ax = plt.subplots()

    aux = lambda_N * ((alpha * (2*beta - alpha))/(2 * (beta - alpha)))

    x = np.linspace(0, 10, 1000)
    ax.plot(x, lambda_N * ((alpha * (2*beta - alpha))/(2 * (beta - alpha))) * np.exp(-(beta - alpha) * x), label="Real cov dens")
    ax.plot(x, lambda_estim * ((alpha_estim * (2*beta_estim - alpha_estim))/(2 * (beta_estim - alpha_estim))) * np.exp(-(beta_estim - alpha_estim) * x),
            label="Estimated cov dens")
    ax.plot(x, (lambda_N + noise) * ((alpha * (2*beta - alpha))/(2 * (beta - alpha))) * np.exp(-(beta - alpha) * x),
            label="Expected cov dens")
    plt.legend()
    plt.show()

