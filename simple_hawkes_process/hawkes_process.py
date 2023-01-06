# Imports
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib
import scipy
from collections import deque
from scipy import stats
import os
from pathlib import Path


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
        # aux_x = np.array([])
        # aux_lambda = np.array([])
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

    def simulate_jumps_every(self, noise, num_first, every=10, random=False):
        self.estimate_max_interval()
        self.max_time = np.infty
        self.simulated = True

        self.every = every
        self.list_of_timestamps = []
        self.list_of_timestamps_type = [[] for i in range(self.M)]
        count = 0
        flag = (count < self.max_jumps)
        while flag:
            # This is not the exact upper-bound but considering the max of the maximal jump
            upper_intensity = self.concurrent_events
            # print(upper_intensity)

            self.t += np.random.exponential(1 / upper_intensity)
            self.update_decks()
            # print(self.intensity_jumps_type[0][-1]+ self.intensity_jumps_type[1][-1])
            # print([max(self.intensity_jumps_type[i][-1], 0) / upper_intensity for i in
            #                                        range(self.M)] + [0])
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

                if count%every == 0:
                    self.list_of_timestamps += [self.timestamps.copy()]
                    for r in range(self.M):
                        self.list_of_timestamps_type[r] += [self.timestamps_type[r].copy()]

                for i in range(self.M):
                    if len(self.decks[i][type_event]) > 0:
                        self.timestamps_type_plot[i] += [self.t]
                        self.decks[i][type_event][0].append(self.t)

                        self.kernel_time[i][type_event] += [self.t]

                        self.kernel_intensity[i][type_event] += [
                            self.kernel_intensity[i][type_event][-1] + self.kernels[i][type_event][1, 0]]
                        self.intensity_jumps_type[i] += [
                            self.intensity_jumps_type[i][-1] + self.kernels[i][type_event][1, 0]]
        if random:
            self.add_noise_random_evolution(noise, num_first)
        else:
            self.add_noise_non_merged_evolution(noise, num_first)

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
