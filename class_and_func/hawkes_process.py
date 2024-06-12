import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy import stats
from collections import deque
import os
from pathlib import Path


class exp_thinning_hawkes(object):
    """
    Univariate Hawkes process with exponential kernel. No events or initial condition before initial time.
    """

    def __init__(self, lambda_0, alpha, beta, t=0.0, max_jumps=None, max_time=None):
        """
        Parameters
        ----------
        lambda_0 : float
            Baseline constant intensity.
        alpha : float
            Interaction factor.
        beta : float
            Decay factor.
        t : float, optional
            Initial time. The default is 0.
        max_jumps : float, optional
            Maximal number of jumps. The default is None.
        max_time : float, optional
            Maximal time horizon. The default is None.
            
        Attributes
        ----------
        t_0 : float
            Initial time provided at initialization.
        timestamps : list of float
            List of simulated events. It includes the initial time t_0.
        intensity_jumps : list of float
            List of intensity at each simulated jump. It includes the baseline intensity lambda_0.
        aux : float
            Parameter used in simulation.
        simulated : bool
            Parameter that marks if a process has been already been simulated, or if its event times have been initialized.
        """
        self.alpha = alpha
        self.beta = beta
        self.t_0 = t
        self.t = t
        self.lambda_0 = lambda_0
        self.max_jumps = max_jumps
        self.max_time = max_time
        self.timestamps = [t]
        self.intensity_jumps = [lambda_0]
        self.aux = 0
        self.simulated = False

    def simulate(self):
        """
        Auxiliary function to check if already simulated and, if not, which simulation to launch.

        Simulation follows Ogata's adapted thinning algorithm.

        Works with both self-exciting and self-regulating processes.
        
        To launch simulation either self.max_jumps or self.max_time must be other than None, so the algorithm knows when to stop.
        """
        if not self.simulated:
            if self.max_jumps is not None and self.max_time is None:
                self.simulate_jumps()
            elif self.max_time is not None and self.max_jumps is None:
                self.simulate_time()
            else:
                print("Either max_jumps or max_time must be given.")
            self.simulated = True

        else:
            print("Process already simulated")

    def simulate_jumps(self):
        """
        Simulation is done until the maximal number of jumps (self.max_jumps) is attained.
        """
        flag = 0

        candidate_intensity = self.lambda_0

        while flag < self.max_jumps:

            upper_intensity = max(self.lambda_0,
                                  self.lambda_0 + self.aux * np.exp(-self.beta * (self.t - self.timestamps[-1])))

            self.t += np.random.exponential(1 / upper_intensity)
            candidate_intensity = self.lambda_0 + self.aux * np.exp(-self.beta * (self.t - self.timestamps[-1]))

            if upper_intensity * np.random.uniform() <= candidate_intensity:
                self.timestamps += [self.t]
                self.intensity_jumps += [candidate_intensity + self.alpha]
                self.aux = candidate_intensity - self.lambda_0 + self.alpha
                flag += 1

        self.max_time = self.timestamps[-1]
        # We have to add a "self.max_time = self.timestamps[-1] at the end so plot_intensity works correctly"

    def simulate_time(self):
        """
        Simulation is done until an event that surpasses the time horizon (self.max_time) appears.
        """
        flag = self.t < self.max_time

        while flag:
            upper_intensity = max(self.lambda_0,
                                  self.lambda_0 + self.aux * np.exp(-self.beta * (self.t - self.timestamps[-1])))

            self.t += np.random.exponential(1 / upper_intensity)
            candidate_intensity = self.lambda_0 + self.aux * np.exp(-self.beta * (self.t - self.timestamps[-1]))

            flag = self.t < self.max_time

            if upper_intensity * np.random.uniform() <= candidate_intensity and flag:
                self.timestamps += [self.t]
                self.intensity_jumps += [candidate_intensity + self.alpha]
                self.aux = self.aux * np.exp(-self.beta * (self.t - self.timestamps[-2])) + self.alpha

    def plot_intensity(self, ax=None, plot_N=True):
        """
        Plot intensity function. If plot_N is True, plots also step function N([0,t]).
        The parameter ax allows to plot the intensity function in a previously created plot.

        Parameters
        ----------
        ax : .axes.Axes or array of Axes, optional.
            If None, method will generate own figure.
            Otherwise, will use given axes. Must be '.axes.Axes' if plot_N = False and array of shape (2,1) if True.
        plot_N : bool, optional.
            Whether we plot the step function N or not.
        """
        if not self.simulated:
            print("Simulate first")

        else:
            if plot_N:
                if ax is None:
                    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                elif isinstance(ax[0], matplotlib.axes.Axes):
                    ax1, ax2 = ax
                else:
                    return "ax must be a (2,1) axes"
            else:
                if ax is None:
                    fig, ax1 = plt.subplots()
                elif isinstance(ax, matplotlib.axes.Axes):
                    ax1 = ax
                else:
                    return "ax must be an instance of an axes"

            self.timestamps.append(self.max_time)

            times = np.array([self.timestamps[0], self.timestamps[1]])
            intensities = np.array([self.lambda_0, self.lambda_0])
            step = 0.01
            for i, lambda_k in enumerate(self.intensity_jumps):
                if i != 0:
                    T_k = self.timestamps[i]
                    nb_step = np.maximum(100, np.floor((self.timestamps[i + 1] - T_k) / step))
                    aux_times = np.linspace(T_k, self.timestamps[i + 1], int(nb_step))
                    times = np.append(times, aux_times)
                    intensities = np.append(intensities, self.lambda_0 + (lambda_k - self.lambda_0) * np.exp(
                        -self.beta * (aux_times - T_k)))

            ax1.plot([0, self.max_time], [0, 0], c='k', alpha=0.5)
            #if self.alpha < 0:
                #ax1.plot(times, intensities, label="Underlying intensity", c="#1f77b4")
            ax1.plot(times, np.maximum(intensities, 0), label="Conditional intensity", c='r')
            ax1.legend()
            ax1.grid()
            if plot_N:
                ax2.step(self.timestamps, np.append(np.arange(0, len(self.timestamps) - 1), len(self.timestamps) - 2),
                         where="post", label="$N(t)$")
                ax2.legend()
                ax2.grid()
            self.timestamps.pop()

    def set_time_intensity(self, timestamps):
    
        """
        Method to initialize a Hawkes process with a given list of timestamps. 
        
        It computes the corresponding intensity with respect to the parameters given at initialization.
        
        Parameters
        ----------
        timestamps : list of float
            Imposed jump times. Intensity is adjusted to this list of times. Must be ordered list of times.
            It is best if obtained by simulating from another instance of Hawkes process.

        """
        
        if not self.simulated:
            self.timestamps = timestamps
            self.max_time = timestamps[-1]

            intensities = [self.lambda_0]
            for k in range(1, len(timestamps)):
                intensities += [self.lambda_0 + (intensities[-1] - self.lambda_0) * np.exp(
                    -self.beta * (timestamps[k] - timestamps[k - 1])) + self.alpha]
            self.intensity_jumps = intensities
            self.simulated = True

        else:
            print("Already simulated")

    def compensator_transform(self, plot=None, exclude_values=0):
        """
        Obtains transformed times for use of goodness-of-fit tests. 
        
        Transformation obtained through time change theorem.

        Parameters
        ----------
        plot : .axes.Axes, optional.
            If None, then it just obtains the transformed times, otherwise plot the Q-Q plot. The default is None
        exclude_values : int, optional.
            If 0 then takes all transformed points in account during plot. Otherwise, excludes first 'exclude_values'
            values from Q-Q plot. The default is 0.
        """

        if not self.simulated:
            print("Simulate first")

        else:

            T_k = self.timestamps[1]

            compensator_k = self.lambda_0 * (T_k - self.t_0)

            self.timestamps_transformed = [compensator_k]
            self.intervals_transformed = [compensator_k]

            for k in range(2, len(self.timestamps)):

                lambda_k = self.intensity_jumps[k-1]
                tau_star = self.timestamps[k] - self.timestamps[k - 1]
                if lambda_k >= 0:
                    C_k = lambda_k - self.lambda_0
                else:
                    C_k = -self.lambda_0
                    tau_star -= (np.log(-(lambda_k - self.lambda_0)) - np.log(self.lambda_0)) / self.beta

                compensator_k = self.lambda_0 * tau_star + (C_k / self.beta) * (1 - np.exp(-self.beta * tau_star))

                self.timestamps_transformed += [self.timestamps_transformed[-1] + compensator_k]
                self.intervals_transformed += [compensator_k]

            if plot is not None:
                stats.probplot(self.intervals_transformed[exclude_values:], dist=stats.expon, plot=plot)


class multi_simple_hawkes(object):
    def __init__(self, baselines, kernels, t=0, max_jumps=None, max_time=None):

        self.baselines = baselines  # Array of shape (M,)
        self.kernels = kernels  # List with M lists of M arrays
        self.t_0 = t
        self.t = t
        self.max_jumps = max_jumps
        self.max_time = max_time

        self.M = len(self.baselines)
        self.supports = np.array([[self.kernels[i][j].shape[1] for j in range(self.M)] for i in range(self.M)])

        self.timestamps = []  # All Event times
        self.timestamps_type = [[] for i in range(self.M)]
        self.timestamps_type_plot = [[] for i in range(self.M)]  # Each type of event time. List of M floats.
        self.kernel_time = [[[] for j in range(self.M)] for i in range(self.M)]
        self.intensity_jumps = [np.sum(self.baselines)]  # Sum of intensities (Total intensity)
        self.intensity_jumps_type = [[self.baselines[i]] for i in
                                     range(self.M)]  # Each type of intensity time. List of M floats.
        self.kernel_intensity = [[[0] for j in range(self.M)] for i in range(self.M)]

        self.decks = [[[deque() for k in range(self.supports[i, j])] for j in range(self.M)] for i in range(self.M)]

        self.concurrent_events = np.linalg.norm(self.baselines, 1)

        self.simulated = False

    def estimate_max_interval(self):
        max_pos = np.max([np.max(self.kernels[i][j][1]) for i in range(self.M) for j in range(self.M)])
        max_pos = np.max(max_pos, 0)
        # print(max_pos)

        # This should be used if trying to determine the exact max upperbound of the interval
        # min_neg = np.min([np.min(self.kernels[i][j][1]) for i in range(self.M) for j in range(self.M)])
        # min_neg = np.min(min_neg, 0)
        min_neg = 0

        self.max_intensity_jump = max_pos - min_neg

    def simulate(self):
        """
        Auxiliary function to check if already simulated and, if not, which simulation to launch.

        Simulation made by Ogata's adapted thinning algorithm.

        Works with both self-exciting and self-regulating processes.
        """
        if not self.simulated:
            if self.max_jumps is not None and self.max_time is None:
                self.estimate_max_interval()
                self.max_time = np.infty
                self.simulate_jumps()
            elif self.max_time is not None and self.max_jumps is None:
                self.estimate_max_interval()
                self.simulate_time()
            else:
                print("Fix either max_jumps or max_time")
            self.simulated = True

        else:
            print("Process already simulated")

    def update_decks(self):
        aux_x = np.array([])
        aux_lambda = np.array([])
        for i in range(self.M):
            aux_x_dim = np.array([])
            aux_lambda_dim = np.array([])
            for j in range(self.M):
                aux_x = np.array([])
                aux_lambda = np.array([])

                for k in range(self.supports[i, j]):
                    while len(self.decks[i][j][k]) != 0 and self.t - self.decks[i][j][k][0] > self.kernels[i][j][0, k] \
                            and self.decks[i][j][k][0] + self.kernels[i][j][0, k] < self.max_time:
                        if k != self.supports[i, j] - 1:
                            self.decks[i][j][k + 1].append(self.decks[i][j][k].popleft())
                            aux_x = np.append(aux_x, self.kernels[i][j][0, k] + self.decks[i][j][k + 1][-1])
                            aux_lambda = np.append(aux_lambda, self.kernels[i][j][1, k + 1] - self.kernels[i][j][1, k])
                        else:
                            aux_x = np.append(aux_x, self.kernels[i][j][0, k] + self.decks[i][j][k].popleft())
                            aux_lambda = np.append(aux_lambda, -self.kernels[i][j][1, k])
                            self.concurrent_events -= self.max_intensity_jump
                            # print("Type event out :", j, "affected to :", i)

                idx = np.argsort(aux_x)
                aux_x = aux_x[idx]
                aux_lambda = aux_lambda[idx]

                aux_x_dim = np.append(aux_x_dim, aux_x)
                aux_lambda_dim = np.append(aux_lambda_dim, aux_lambda)

                self.kernel_time[i][j] += list(aux_x)
                for l in aux_lambda:
                    self.kernel_intensity[i][j] += [self.kernel_intensity[i][j][-1] + l]

            idx = np.argsort(aux_x_dim)
            aux_x_dim = aux_x_dim[idx]
            aux_lambda_dim = aux_lambda_dim[idx]

            self.timestamps_type_plot[i] += list(aux_x_dim)
            for l in aux_lambda_dim:
                self.intensity_jumps_type[i] += [self.intensity_jumps_type[i][-1] + l]

    def simulate_time(self):
        flag = self.t < self.max_time
        while flag:
            # This is not the exact upper-bound but considering the max of the maximal jump
            upper_intensity = self.concurrent_events

            self.t += np.random.exponential(1 / upper_intensity)
            self.update_decks()
            flag = self.t < self.max_time
            type_event = np.random.multinomial(1, [max(self.intensity_jumps_type[i][-1], 0) / upper_intensity for i in
                                                   range(self.M)] + [0]).argmax()

            if type_event < self.M and flag:
                # print("Type event in: ", type_event)
                # Add to general list
                self.timestamps += [self.t]
                # Add to respective list of timestamps and update respective decks and intensities.
                self.timestamps_type[type_event] += [self.t]
                self.concurrent_events += self.M * self.max_intensity_jump

                for i in range(self.M):
                    if len(self.decks[i][type_event]) > 0:
                        self.timestamps_type_plot[i] += [self.t]
                        self.decks[i][type_event][0].append(self.t)

                        self.kernel_time[i][type_event] += [self.t]

                        self.kernel_intensity[i][type_event] += [
                            self.kernel_intensity[i][type_event][-1] + self.kernels[i][type_event][1, 0]]
                        self.intensity_jumps_type[i] += [
                            self.intensity_jumps_type[i][-1] + self.kernels[i][type_event][1, 0]]
                # print("")

    def simulate_jumps(self):
        count = 0
        flag = (count < self.max_jumps)
        while flag:
            # This is not the exact upper-bound but considering the max of the maximal jump
            upper_intensity = self.concurrent_events

            self.t += np.random.exponential(1 / upper_intensity)
            self.update_decks()
            type_event = np.random.multinomial(1, [max(self.intensity_jumps_type[i][-1], 0) / upper_intensity for i in
                                                   range(self.M)] + [0]).argmax()

            if type_event < self.M:
                # print("Type event in: ", type_event)
                # Add to general list
                self.timestamps += [self.t]
                # Add to respective list of timestamps and update respective decks and intensities.
                self.timestamps_type[type_event] += [self.t]
                self.concurrent_events += self.M * self.max_intensity_jump

                count += 1
                flag = (count < self.max_jumps)

                for i in range(self.M):
                    if len(self.decks[i][type_event]) > 0:
                        self.timestamps_type_plot[i] += [self.t]
                        self.decks[i][type_event][0].append(self.t)

                        self.kernel_time[i][type_event] += [self.t]

                        self.kernel_intensity[i][type_event] += [
                            self.kernel_intensity[i][type_event][-1] + self.kernels[i][type_event][1, 0]]
                        self.intensity_jumps_type[i] += [
                            self.intensity_jumps_type[i][-1] + self.kernels[i][type_event][1, 0]]
                # print("")

    def plot_intensity(self, plot_candidates=False):
        if not self.simulated:
            print("Simulate first")
        else:
            fig, ax = plt.subplots(self.M + 1, self.M + 1, sharex="col")
            plt.rcParams['lines.linewidth'] = 1

            self.t = self.max_time
            self.update_decks()

            cmap = cm.get_cmap('viridis')
            colors = [cmap(i / self.M) for i in range(self.M)]

            for i in range(self.M):

                self.intensity_jumps_type[i] += [self.intensity_jumps_type[i][-1]]

                x_type = [self.t_0] + [i for i in self.timestamps_type_plot[i] for j in range(2)] + [self.max_time]
                y_type = [i for i in self.intensity_jumps_type[i][:-1] for j in range(2)]
                y_pos_type = [max(i, 0) for i in self.intensity_jumps_type[i][:-1] for j in range(2)]

                ax[i, self.M].plot(x_type, y_pos_type, label="real intensity")
                ax[i, self.M].plot(x_type, y_type, label="underlying intensity")

                for k in self.timestamps_type[i]:
                    ax[self.M, i].plot([k, k], [0, 1], c=colors[i], alpha=0.5, linestyle="--")

                for j in range(self.M):
                    self.kernel_intensity[i][j] += [self.kernel_intensity[i][j][-1]]

                    x = [self.t_0] + [i for i in self.kernel_time[i][j] for j in range(2)] + [self.max_time]
                    y = [i for i in self.kernel_intensity[i][j][:-1] for j in range(2)]

                    ax[i, j].plot(x, y, label="underlying intensity")

            # ax[self.M - 1, self.M].legend()

            return "Done"

    def plot_events(self, plot_candidates=False):
        if not self.simulated:
            print("Simulate first")
        else:
            self.t = self.max_time
            self.update_decks()
            fig, ax = plt.subplots()
            for i in self.timestamps:
                ax.plot([i, i], [0.75, 1], c="k", linewidth=0.5)
            for i in self.timestamps_type_plot[0]:
                ax.plot([i, i], [0.5, 0.75], c="r", linewidth=0.5)
            for i in self.timestamps_type_plot[1]:
                ax.plot([i, i], [0.25, 0.5], c="b", linewidth=0.5)

    def add_nmc(self, noise, write=False, num_first=0):
        self.noise = noise
        noised_intervals = []
        for i, aux_list in enumerate(self.timestamps_type):
            aux_deck = deque(aux_list)
            u = np.random.uniform(-self.noise, 0)
            noised_list = [aux_deck.popleft() + u]
            right_limit = noised_list[0] + self.noise
            noised_limit = []

            while len(aux_deck) > 1:
                aux = aux_deck.popleft()
                u = np.random.uniform(-self.noise, 0)
                if aux <= right_limit:
                    pass
                elif aux + u <= right_limit:
                    right_limit = aux + u + self.noise
                else:
                    noised_limit += [right_limit]
                    noised_list += [aux + u]
                    right_limit = aux + u + self.noise

            noised_limit += [right_limit]
            if write:
                path = "noised_intervals_data/" + str(self.M) + "Eta" + str(self.noise)
                try:
                    os.mkdir(path)
                except OSError:
                    pass
                else:
                    print("Successfully created the directory %s " % path)

                path = path + "/"

                f_real = open(Path(path + "HawkesRealPoint" + str(num_first*100 + i) + ".bed"), 'w')
                f_noised = open(Path(path + "HawkesNoisedPoint" + str(num_first*100 + i) + ".bed"), 'w')
                f_interval = open(Path(path + "HawkesNoisedInterval" + str(num_first*100 + i) + ".bed"), 'w')
                for j, k, l in zip(aux_list, noised_list, noised_limit):
                    mid = (j+k)/2
                    f_real.write("chr1" + "\t" + str(j) + "\t" + str(j) + "\n")
                    f_noised.write("chr1" + "\t" + str(mid) + "\t" + str(mid) + "\n")
                    f_interval.write("chr1" + "\t" + str(k) + "\t" + str(l) + "\n")

    def add_nmc_check(self, noise):
        fig, ax = plt.subplots(3, 2, sharex=True)
        self.noise = noise
        noised_intervals = []
        for i, aux_list in enumerate(self.timestamps_type):
            aux_deck = deque(aux_list)
            u = np.random.uniform(-self.noise, 0)
            noised_list = [aux_deck.popleft() + u]
            bla = [noised_list[0]]
            right_limit = noised_list[0] + self.noise
            noised_limit = []

            while len(aux_deck) > 1:
                aux = aux_deck.popleft()
                u = np.random.uniform(-self.noise, 0)
                bla += [aux + u]
                if aux <= right_limit:
                    pass
                elif aux + u <= right_limit:
                    right_limit = aux + u + self.noise
                else:
                    noised_limit += [right_limit]
                    noised_list += [aux + u]
                    right_limit = aux + u + self.noise

            noised_limit += [right_limit]

            print(len(noised_list), len(noised_limit))

            ax[0, i].scatter(aux_list, 0 * np.array(aux_list), s=1)
            ax[0, i].scatter(bla, 0 * np.array(bla)-0.02, s=1)
            ax[1, i].scatter(noised_list, 0*np.array(noised_list), s=1)
            ax[2, i].scatter(noised_list, 0 * np.array(noised_list), s=2, marker=5)
            ax[2, i].scatter(noised_limit, 0 * np.array(noised_limit), s=2, marker=4)
