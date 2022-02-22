import numpy as np
import time
from matplotlib import pyplot as plt
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.likelihood_functions import multivariate_lstsquares_simplified, \
    multivariate_loglikelihood_simplified
from class_and_func.estimator_class import multivariate_estimator_bfgs, multivariate_estimator_bfgs_grad


def multivariate_loglikelihood_repam(theta, tList, dim=None, dimensional=False):
    """

    Parameters
    ----------
    theta : tuple of array
        Tuple containing 3 arrays. First corresponds to vector of baseline intensities mu. Second is a square matrix
        corresponding to interaction matrix alpha. Last is vector of recovery rates beta.

    tList : list of tuple
        List containing tuples (t, m) where t is the time of event and m is the mark (dimension). The marks must go from
        1 to nb_of_dimensions.
        Important to note that this algorithm expects the first and last time to mark the beginning and
        the horizon of the observed process. As such, first and last marks must be equal to 0, signifying that they are
        not real event times.
        The algorithm checks by itself if this condition is respected, otherwise it sets the beginning at 0 and the end
        equal to the last time event.
    dim : int
        Number of processes only necessary if providing 1-dimensional theta. Default is None
    dimensional : bool
        Whether to return the sum of loglikelihood or decomposed in each dimension. Default is False.
Returns
    -------
    likelihood : array of float
        Value of likelihood at each process.
        The value returned is the opposite of the mathematical likelihood in order to use minimization packages.
    """

    if isinstance(theta, np.ndarray):
        if dim is None:
            raise ValueError("Must provide dimension to unpack correctly")
        else:
            mu = np.array(theta[:dim]).reshape((dim, 1))
            alpha = np.array(theta[dim:dim * (dim + 1)]).reshape((dim, dim))
            beta = np.array(theta[dim * (dim + 1):]).reshape((dim, 1))
    else:
        mu, alpha, beta = (i.copy() for i in theta)
    beta = beta + 1e-10

    beta_1 = 1/beta

    timestamps = tList.copy()

    # We first check if we have the correct beginning and ending.
    if timestamps[0][1] > 0:
        timestamps = [(0, 0)] + timestamps
    if timestamps[-1][1] > 0:
        timestamps += [(timestamps[-1][0], 0)]

    # Initialise values
    tb, mb = timestamps[1]
    # Compensator between beginning and first event time
    compensator = mu*(tb - timestamps[0][0])
    # Intensity before first jump
    log_i = np.zeros((alpha.shape[0],1))
    log_i[mb-1] = np.log(mu[mb-1])
    ic = mu + alpha[:, [mb - 1]]*beta
    # j=1

    for tc, mc in timestamps[2:]:
        # First we estimate the compensator
        inside_log = (mu - np.minimum(ic, 0))/mu
        # Restart time
        t_star = tb + np.multiply(beta_1, np.log(inside_log))

        aux = 1/inside_log  # inside_log can't be equal to zero (coordinate-wise)
        aux = np.minimum(1, aux)
        compensator += (t_star < tc)*(np.multiply(mu, tc-t_star) + np.multiply(beta_1, ic-mu)*(aux - np.exp(-beta*(tc-tb))))

        # Then, estimation of intensity before next jump.
        if mc > 0:
            ic = mu + np.multiply((ic - mu), np.exp(-beta*(tc-tb)))

            if ic[mc - 1] <= 0.0:
                # print("oh no")
                # res = 1e8

                # rayon_spec = np.max(np.abs(np.linalg.eig(np.abs(alpha) / beta)[0]))
                # rayon_spec = min(rayon_spec, 0.999)
                # if 2*(tList[-1][0])/(1-rayon_spec) < 0:
                #     print("wut")
                # res = 2*(tList[-1][0])/(1-rayon_spec)

                res = 1e8 #*(np.sum(mu**2) + np.sum(alpha**2) + np.sum(beta**2))
                return res
            else:
                log_i[mc-1] += np.log(ic[mc - 1])
            # j += 1
            # print(j, ic, 1+0.45*(1-0.5**(j-1)))
            ic += alpha[:, [mc - 1]]*beta

        tb = tc
    likelihood = log_i - compensator
    if not(dimensional):
        likelihood = np.sum(likelihood)
    return -likelihood


if __name__ == "__main__":
    np.random.seed(0)

    dim = 2  # 2, 3 ou 4

    mu = 1 * np.array([[1.5], [2.5]])
    alpha = 1 * np.array([[0.0, 0.0], [-1.2, -1.5]])
    beta = 1 * np.array([[1.], [2.]])
    max_jumps = 10000

    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=max_jumps)

    print("Starting simulation...")
    hawkes.simulate()
    print("Finished simulation")
    print("#" * 200)
    print("My real loglikelihood", multivariate_loglikelihood_simplified((mu, alpha, beta), hawkes.timestamps))

    # loglikelihood_estimation = multivariate_estimator_bfgs(dimension=dim, options={"disp": False})
    # print("Starting loglikelihood...")
    # start_time = time.time()
    # loglikelihood_estimation.fit(hawkes.timestamps)
    # end_time = time.time() - start_time
    #
    # print("Estimation through loglikelihood: ", np.round(loglikelihood_estimation.res.x, 3), "\nIn: ", end_time)
    # print("")
    #
    loglikelihood_estimation = multivariate_estimator_bfgs_grad(dimension=dim, options={"disp": False})
    print("Starting grad...")
    start_time = time.time()
    loglikelihood_estimation.fit(hawkes.timestamps)
    end_time = time.time() - start_time

    print("Estimation through loglikelihood: ", np.round(loglikelihood_estimation.res.x, 3), "\nIn: ", end_time)
    print("")

    loglikelihood_estimation = multivariate_estimator_bfgs(loss=multivariate_loglikelihood_repam, dimension=dim, options={"disp": False})
    print("Starting repam...")
    start_time = time.time()
    loglikelihood_estimation.fit(hawkes.timestamps)
    end_time = time.time() - start_time
    print("Estimation through loglikelihood: ", np.round(loglikelihood_estimation.res.x, 3), "\nIn: ", end_time)