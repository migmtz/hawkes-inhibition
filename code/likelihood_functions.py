# Imports
import numpy as np


# Functions for exponential estimation of loglikelihood.

def loglikelihood(theta, tList):
    """
    Exact computation of the loglikelihood for an exponential Hawkes process for either self-exciting or self-regulating cases. 
    Estimation for a single realization.
    
    Parameters
    ----------
    theta : tuple of float
        Tuple containing the parameters to use for estimation.
    tList : list of float
        List containing all the lists of data (event times).

    Returns
    -------
    likelihood : float
        Value of likelihood, either for 1 realization or for a batch. 
        The value returned is the opposite of the mathematical likelihood in order to use minimization packages.
    """
    # Extract variables
    lambda0, alpha, beta = theta

    # Avoid wrong values in algorithm such as negative lambda0 or beta
    if lambda0 <= 0 or beta <= 0:
        return 1e5

    else:

        compensator_k = lambda0 * tList[1]
        lambda_avant = lambda0
        lambda_k = lambda0 + alpha

        if lambda_avant <= 0:
            return 1e5

        likelihood = np.log(lambda_avant) - compensator_k

        # Iteration
        for k in range(2, len(tList)):

            if lambda_k >= 0:
                C_k = lambda_k - lambda0
                tau_star = tList[k] - tList[k - 1]
            else:
                C_k = -lambda0
                tau_star = tList[k] - tList[k - 1] - (np.log(-(lambda_k - lambda0)) - np.log(lambda0)) / beta

            lambda_avant = lambda0 + (lambda_k - lambda0) * np.exp(-beta * (tList[k] - tList[k - 1]))
            lambda_k = lambda_avant + alpha
            compensator_k = lambda0 * tau_star + (C_k / beta) * (1 - np.exp(-beta * tau_star))

            if lambda_avant <= 0:
                return 1e5

            likelihood += np.log(lambda_avant) - compensator_k

        # We return the opposite of the likelihood in order to use minimization packages.
        return -likelihood


def likelihood_approximated(theta, tList):
    """
    Approximation method for the loglikelihood, proposed by Lemonnier.
    Estimation for a single realization.

    Parameters
    ----------
    theta : tuple of float
        Tuple containing the parameters to use for estimation.
    tList : list of float
        List containing all the lists of data (event times).

    Returns
    -------
    likelihood : float
        Value of likelihood, either for 1 realization or for a batch.
        The value returned is the opposite of the mathematical likelihood in order to use minimization packages.
    """
    lambda0, alpha, beta = theta

    # Avoid wrong values in algorithm such as negative lambda0 or beta
    if lambda0 <= 0 or beta <= 0:
        return 1e5

    else:
        # Auxiliary values
        aux = np.log(lambda0)  # Value that will be often used

        # Set initial values and first step of iteration
        A_k_minus = 0
        Lambda_k = 0
        # likelihood = - lambda0*tList[0] + np.log(A_k_minus + lambda0)
        likelihood = - lambda0 * tList[-1] + np.log(A_k_minus + lambda0)
        tLast = tList[1]

        # Iteration
        for k in range(2, len(tList)):

            # Update A(k)
            tNext = tList[k]
            tau_k = tNext - tLast
            A_k = (A_k_minus + alpha)

            # Integral
            Lambda_k = (A_k / beta) * (1 - np.exp(-beta * tau_k))  # + lambda0*tau_k

            # Update likelihood

            A_k_minus = A_k * np.exp(-beta * tau_k)
            if A_k_minus + lambda0 <= 0:
                return 1e5
            likelihood = likelihood - Lambda_k + np.log(lambda0 + A_k_minus)

            # Update B(k) and tLast

            tLast = tNext

        # We return the opposite of the likelihood in order to use minimization packages.
        return -likelihood


def batch_likelihood(theta, nList, exact=True, penalized=False, C=1):
    """
    Wrapper function that allows to call either the exact or penalized loglikelihood functions aswell as an L2-penalization.
    
    This function works either with 1 or multiple (batch) realizations of Hawkes process.
    
    Parameters
    ----------
    theta : tuple of float
        Tuple containing the parameters to use for estimation.
    nList : list of list of float
        List containing all the lists of data (event times).
    exact : bool
        Whether to use the exact computation method (True) or the approximation by Lemonnier. Default is True.
    penalized : bool
        Whether to add an L2-penalization. Default is False.
    C : float
        Penalization factor, only used if penalized parameter is True. Default is 1.

    Returns
    -------
    batchlikelihood : float
        Value of likelihood, either for 1 realization or for a batch.
    """
    batchlikelihood = 0

    if exact:
        func = lambda x, y: loglikelihood(x, y)
    else:
        func = lambda x, y: likelihood_approximated(x, y)

    for tList in nList:
        batchlikelihood += func(theta, tList)
    batchlikelihood /= len(nList)

    if penalized:
        batchlikelihood += C * (theta[0] ** 2 + theta[1] ** 2 + theta[2] ** 2)

    return batchlikelihood


def multivariate_loglikelihood_simplified(theta, tList):
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

    Returns
    -------
    likelihood : array of float
        Value of likelihood at each process.
        The value returned is the opposite of the mathematical likelihood in order to use minimization packages.
    """

    mu, alpha, beta = (i.copy() for i in theta)
    beta[beta == 0] = 1

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

    ic = mu + alpha[:, [mb - 1]]

    for tc, mc in timestamps[2:]:
        # First we estimate the compensator
        # print(mu.shape, ic.shape, alpha[:,mb-1].shape)
        inside_log = (mu - np.minimum(ic, 0))/mu
        # Restart time
        t_star = tb + np.multiply(beta_1, np.log(inside_log))
        # print(inside_log.shape, beta_1.shape, t_star.shape)
        aux = inside_log
        aux[aux == 0] = 1

        compensator += (t_star < tc)*(np.multiply(mu, tc-t_star) + np.multiply(beta_1, ic-mu)*(aux - np.exp(-beta*(tc-tb))))

        # Then, estimation of intensity before next jump.
        if mc > 0:
            ic = mu + np.multiply((ic - mu), np.exp(-beta*(tc-tb)))

            if ic[mc - 1] <= 0.0:
                return np.inf
            else:
                log_i[mc-1] += np.log(ic[mc - 1])

            ic += alpha[:, [mc - 1]]

        tb = tc
    print(log_i.shape, compensator.shape)
    likelihood = log_i - compensator
    return -likelihood
