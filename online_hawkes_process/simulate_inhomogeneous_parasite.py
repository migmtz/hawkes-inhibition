from class_and_func.hawkes_process import exp_thinning_hawkes
from class_and_func.estimator_class import loglikelihood_estimator_bfgs
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import vonmises


def sinusoidal(x, lam):
    return lam*(np.sin(x) + 1)/2


class inhomogeneous_poisson_process(object):
    """
    Univariate Hawkes process with exponential kernel. No events or initial condition before initial time.
    """

    def __init__(self, f, t=0.0, max_jumps=None, max_time=None):
        """
        """
        self.f = f
        self.t_0 = t
        self.t = t
        self.max_jumps = max_jumps
        self.max_time = max_time
        self.timestamps = [t]
        self.simulated = False

    def simulate(self, upper_bound, plot=False):
        """
        Auxiliary function to check if already simulated and, if not, which simulation to launch.

        Simulation follows Ogata's adapted thinning algorithm.

        Works with both self-exciting and self-regulating processes.

        To launch simulation either self.max_jumps or self.max_time must be other than None, so the algorithm knows when to stop.
        """
        if not self.simulated:
            if self.max_jumps is not None and self.max_time is None:
                self.simulate_jumps(upper_bound)
            elif self.max_time is not None and self.max_jumps is None:
                self.simulate_time(upper_bound, plot)
            else:
                print("Either max_jumps or max_time must be given.")
            self.simulated = True

        else:
            print("Process already simulated")

    def simulate_jumps(self, upper_bound):
        """
        Simulation is done until the maximal number of jumps (self.max_jumps) is attained.
        """
        flag = 0

        while flag < self.max_jumps:

            self.t += np.random.exponential(1 / upper_bound)
            candidate_intensity = self.f(self.t)

            if upper_bound * np.random.uniform() <= candidate_intensity:
                self.timestamps += [self.t]
                flag += 1

        self.max_time = self.timestamps[-1]
        # We have to add a "self.max_time = self.timestamps[-1] at the end so plot_intensity works correctly"

    def simulate_time(self, upper_bound, plot=False):
        """
        Simulation is done until an event that surpasses the time horizon (self.max_time) appears.
        """
        flag = self.t < self.max_time
        if plot:
            fig, ax = plt.subplots()
            x = np.linspace(0, self.max_time, 10000)
            ax.plot(x, self.f(x), label="intensity", c="r")

        while flag:
            self.t += np.random.exponential(1 / upper_bound)
            candidate_intensity = self.f(self.t)

            flag = self.t < self.max_time

            if plot:
                ax.scatter(self.t, candidate_intensity, c="b")

            if upper_bound * np.random.uniform() <= candidate_intensity and flag:
                self.timestamps += [self.t]
                if plot:
                    ax.scatter(self.t, candidate_intensity, c="r")

        if plot:
            ax.set_xlim([-0.05, self.max_time+0.05])
            plt.grid()
            plt.show()

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

    f = vonmises.pdf
    np.random.seed(0)

    for i, noise in enumerate(np.linspace(0.05, 10, steps)):
        upper_bound = f(0, noise)

        for j in range(repet):
            ppp = inhomogeneous_poisson_process(f=lambda x: f(x, noise), max_time= 50)
            ppp.simulate(upper_bound=upper_bound, plot=False)

            parasited_times = [0.0] + np.sort(hp.timestamps[1:] + ppp.timestamps[1:]).tolist()

            paras_estimation = loglikelihood_estimator_bfgs(initial_guess=np.ones(3))
            paras_estimation.fit(parasited_times)

            estimated_list[i, :, j] += np.array(paras_estimation.res.x)

    aux = np.swapaxes(np.array([np.ones((steps,repet))*mu, np.ones((steps,repet))*alpha, np.ones((steps,repet))*beta]), 0, 1)
    estimated_list = np.abs((estimated_list - aux)/aux)
    print("")
    fig_est, ax_est = plt.subplots(3, 1, sharex=True)

    smooth = 7

    #ax_est[0].plot([0.05, 2], [mu, mu])
    #ax_est[0].plot([0.05, 2], [real_estimation.res.x[0], real_estimation.res.x[0]])
    #sns.violinplot(data=estimated_list[:, 0, :].T, ax=ax_est[0])
    ax_est[0].scatter(range(0, steps), np.mean(estimated_list[:, 0, :], axis=1))
    aux = np.convolve(np.mean(estimated_list[:, 0, :], axis=1), (1 / smooth) * np.ones(smooth))
    aux2 = [np.mean(np.mean(estimated_list[:, 0, :], axis=1)[:i + 1]) for i in range(0, smooth)]
    ax_est[0].plot(range(smooth), aux2, c="r", alpha=0.8)
    ax_est[0].plot(range(smooth - 1, len(aux) - smooth), aux[smooth - 1:-smooth], c="r", alpha=0.8)
    ax_est[0].set_xticklabels([])

    #ax_est[1].plot([0.05, 2], [0.5, 0.5])
    #ax_est[1].plot([0.05, 2], [real_estimation.res.x[1], real_estimation.res.x[1]])
    #sns.violinplot(data=estimated_list[:, 1, :].T, ax=ax_est[1])
    ax_est[1].scatter(range(0, steps), np.mean(estimated_list[:, 1, :], axis=1))
    aux = np.convolve(np.mean(estimated_list[:, 1, :], axis=1), (1/smooth) * np.ones(smooth))
    aux2 = [np.mean(np.mean(estimated_list[:, 1, :], axis=1)[:i+1]) for i in range(0, smooth)]
    ax_est[1].plot(range(smooth), aux2, c="r", alpha=0.8)
    ax_est[1].plot(range(smooth-1, len(aux)-smooth), aux[smooth-1:-smooth], c="r", alpha=0.8)
    ax_est[1].set_xticklabels([])

    #ax_est[2].plot([0.05, 2], [beta, beta])
    #ax_est[2].plot([0.05, 2], [real_estimation.res.x[2], real_estimation.res.x[2]])
    #sns.violinplot(data=estimated_list[:, 2, :].T, ax=ax_est[2])
    ax_est[2].scatter(range(0, steps), np.mean(estimated_list[:, 2, :], axis=1))
    aux = np.convolve(np.mean(estimated_list[:, 2, :], axis=1), (1 / smooth) * np.ones(smooth))
    aux2 = [np.mean(np.mean(estimated_list[:, 2, :], axis=1)[:i + 1]) for i in range(0, smooth)]
    ax_est[2].plot(range(smooth), aux2, c="r", alpha=0.8)
    ax_est[2].plot(range(smooth - 1, len(aux) - smooth), aux[smooth - 1:-smooth], c="r", alpha=0.8)

    ax_est[2].set_xticks(np.arange(0, steps, 2))
    ax_est[2].set_xticklabels([np.round(0.1*k+0.05, 2) for k in range(20)])
    plt.show()