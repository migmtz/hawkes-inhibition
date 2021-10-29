# Imports
import numpy as np
import torch
import tensorflow as tf


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


def multivariate_loglikelihood_simplified(theta, tList, dim=None, dimensional=False):
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
    # j=1

    for tc, mc in timestamps[2:]:
        # First we estimate the compensator
        inside_log = (mu - np.minimum(ic, 0))/mu
        # Restart time
        t_star = tb + np.multiply(beta_1, np.log(inside_log))
        aux = inside_log
        aux[aux == 0] = 1

        compensator += (t_star < tc)*(np.multiply(mu, tc-t_star) + np.multiply(beta_1, ic-mu)*(aux - np.exp(-beta*(tc-tb))))

        # Then, estimation of intensity before next jump.
        if mc > 0:
            ic = mu + np.multiply((ic - mu), np.exp(-beta*(tc-tb)))

            if ic[mc - 1] <= 0.0:
                # print("oh no", ic,mc, tc)
                # res = 1e8

                # rayon_spec = np.max(np.abs(np.linalg.eig(np.abs(alpha) / beta)[0]))
                # rayon_spec = min(rayon_spec, 0.999)
                # if 2*(tList[-1][0])/(1-rayon_spec) < 0:
                #     print("wut")
                # res = 2*(tList[-1][0])/(1-rayon_spec)

                res = 1e8*(np.sum(mu**2) + np.sum(alpha**2) + np.sum(beta**2))
                return res
            else:
                log_i[mc-1] += np.log(ic[mc - 1])
            # j += 1
            # print(j, ic, 1+0.45*(1-0.5**(j-1)))
            ic += alpha[:, [mc - 1]]

        tb = tc
    likelihood = log_i - compensator
    if not(dimensional):
        likelihood = np.sum(likelihood)
    return -likelihood


def multivariate_lstsquares_simplified(theta, tList, dim=None):
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
Returns
    -------
    least_squares_error : array of float
        Value of least-squares error at each process.
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
    compensator_sq = (tb - timestamps[0][0])*np.sum(mu**2)
    # Intensity before first jump
    lambda_i = np.zeros((alpha.shape[0],1))
    lambda_i[mb-1] = lambda_i[mb-1] + mu[mb-1]
    ic = mu + alpha[:, [mb - 1]]
    # j=1

    for tc, mc in timestamps[2:]:
        # First we estimate the compensator
        inside_log = (mu - np.minimum(ic, 0))/mu
        # Restart time
        t_star = tb + np.multiply(beta_1, np.log(inside_log))
        # As we consider the cross-interactions in the integral.
        t_star_ij = np.maximum(t_star, t_star.T)

        first_term = (mu*mu.T)*(tc - t_star_ij)
        middle_term = (mu*((ic - mu)*beta_1).T)*(np.exp(-beta.T*(t_star_ij - tb)) - np.exp(-beta.T*(tc - tb)))
        # if j == 1:
        #     print(middle_term)
        #     j += 1
        middle_term = middle_term + middle_term.T
        last_term = (ic - mu)*(ic - mu).T*(np.exp(-(beta + beta.T)*(t_star_ij - tb)) - np.exp(-(beta + beta.T)*(tc - tb)))

        aux = (t_star_ij < tc)*(first_term + middle_term + last_term)

        compensator_sq += np.sum(aux)

        # Then, estimation of intensity before next jump.
        if mc > 0:
            ic = mu + np.multiply((ic - mu), np.exp(-beta*(tc-tb)))

            lambda_i[mc-1] += np.maximum(ic[mc - 1], 0)
            # j += 1
            # print(j, ic, 1+0.45*(1-0.5**(j-1)))
            ic += alpha[:, [mc - 1]]

        tb = tc
    least_squares_error = compensator_sq - (2/timestamps[-1][0])*np.sum(lambda_i)
    return least_squares_error


def multivariate_loglikelihood_torch(theta, tList, dimensional=False):
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
    dimensional : bool
        Whether to return the sum of loglikelihood or decomposed in each dimension. Default is False.
Returns
    -------
    likelihood : array of float
        Value of likelihood at each process.
        The value returned is the opposite of the mathematical likelihood in order to use minimization packages.
    """

    mu, alpha, beta = theta

    mu_alt = torch.exp(mu)
    beta_alt = torch.exp(beta)

    timestamps = tList.copy()

    # We first check if we have the correct beginning and ending.
    if timestamps[0][1] > 0:
        timestamps = [(0, 0)] + timestamps
    if timestamps[-1][1] > 0:
        timestamps += [(timestamps[-1][0], 0)]

    # Initialise values
    tb, mb = timestamps[1]
    # Compensator between beginning and first event time
    compensator = mu_alt * (tb - timestamps[0][0])
    # Intensity before first jump
    log_i = torch.zeros(alpha.shape[0], 1)
    log_i[mb - 1] = torch.log(mu_alt[mb - 1])
    ic = mu_alt + alpha[:, [mb - 1]]
    # j=1

    for tc, mc in timestamps[2:]:
        # First we estimate the compensator
        inside_log = (mu_alt - torch.minimum(ic, torch.tensor(0)))/mu_alt
        # Restart time
        # t_star = tb + torch.log(inside_log)/beta
        aux = inside_log
        aux[aux == 0] = 1

        compensator += ((tb + torch.log(inside_log)/beta_alt) < tc)*(mu_alt*(tc-tb - torch.log(inside_log)/beta_alt)) + ((ic-mu_alt)/beta_alt)*(aux - torch.exp(-beta_alt*(tc-tb)))

        # Then, estimation of intensity before next jump.
        if mc > 0:
            ic = mu_alt + (ic - mu_alt) * torch.exp(-beta_alt * (tc - tb))

            if ic[mc - 1] <= 0.0:
                # print("oh no", ic,mc, tc)
                # res = 1e8

                # rayon_spec = np.max(np.abs(np.linalg.eig(np.abs(alpha) / beta)[0]))
                # rayon_spec = min(rayon_spec, 0.999)
                # if 2*(tList[-1][0])/(1-rayon_spec) < 0:
                #     print("wut")
                # res = 2*(tList[-1][0])/(1-rayon_spec)

                res = 1e8 + (
                            torch.sum((mu_alt + 1) ** 2) + torch.sum((alpha + 1) ** 2) + torch.sum((beta_alt + 1) ** 2))
                return res
            else:
                log_i[mc - 1] = log_i[mc - 1] + torch.log(ic[mc - 1])
            # j += 1
            # print(j, ic, 1+0.45*(1-0.5**(j-1)))
            ic = ic + alpha[:, [mc - 1]]

        tb = tc
    likelihood = log_i - compensator
    if not (dimensional):
        likelihood = torch.sum(likelihood)
    return -likelihood


def multivariate_lstsquares_torch(theta, tList, dimensional=False):
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
    dimensional : bool
        Whether to return the sum of loglikelihood or decomposed in each dimension. Default is False.
Returns
    -------
    likelihood : array of float
        Value of likelihood at each process.
        The value returned is the opposite of the mathematical likelihood in order to use minimization packages.
    """

    mu, alpha, beta = theta

    mu_alt = torch.exp(mu)
    beta_alt = torch.exp(beta)

    timestamps = tList.copy()

    # We first check if we have the correct beginning and ending.
    if timestamps[0][1] > 0:
        timestamps = [(0, 0)] + timestamps
    if timestamps[-1][1] > 0:
        timestamps += [(timestamps[-1][0], 0)]

    # Initialise values
    tb, mb = timestamps[1]
    # Compensator between beginning and first event time
    compensator_sq = (tb - timestamps[0][0]) * torch.sum(mu_alt ** 2)
    # Intensity before first jump
    lambda_i = torch.zeros(alpha.shape[0], 1)
    lambda_i[mb - 1] = lambda_i[mb - 1] + mu_alt[mb - 1]
    ic = mu_alt + alpha[:, [mb - 1]]
    # j=1

    for tc, mc in timestamps[2:]:

        # First we estimate the compensator
        inside_log = (mu_alt - torch.minimum(ic, torch.tensor(0)))/mu_alt
        # Restart time
        t_star = tb + torch.log(inside_log)/beta_alt
        t_star_ij = torch.maximum(t_star, t_star.T)

        first_term = (mu*mu.T)*(tc - t_star_ij)
        middle_term = (mu*((ic - mu)/beta_alt).T)*(torch.exp(-beta_alt.T*(t_star_ij - tb)) - torch.exp(-beta_alt.T*(tc - tb)))
        # if j == 1:
        #     print(middle_term)
        #     j += 1
        middle_term = middle_term + middle_term.T
        last_term = (ic - mu)*(ic - mu).T*(torch.exp(-(beta_alt + beta_alt.T)*(t_star_ij - tb)) - torch.exp(-(beta_alt + beta_alt.T)*(tc - tb)))

        aux = (t_star_ij < tc) * (first_term + middle_term + last_term)

        compensator_sq = compensator_sq + torch.sum(aux)

        if mc > 0:
            ic = mu_alt + (ic - mu_alt) * torch.exp(-beta_alt * (tc - tb))

            lambda_i[mc - 1] = lambda_i[mc - 1] + torch.maximum(ic[mc - 1], torch.tensor(0))
            # j += 1
            # print(j, ic, 1+0.45*(1-0.5**(j-1)))
            ic = ic + alpha[:, [mc - 1]]

        tb = tc

    least_squares_error = compensator_sq - (2 / timestamps[-1][0] * torch.sum(lambda_i))

    return least_squares_error


def multivariate_loglikelihood_tf(theta, tList, dimensional=False):
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
    dimensional : bool
        Whether to return the sum of loglikelihood or decomposed in each dimension. Default is False.
Returns
    -------
    likelihood : array of float
        Value of likelihood at each process.
        The value returned is the opposite of the mathematical likelihood in order to use minimization packages.
    """

    mu, alpha, beta = theta
    dim = mu.shape[0]

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
    log_i = tf.zeros(alpha.shape[0], 1)
    log_i = log_i*np.array([[i==(mb-1)] for i in range(dim)]) + tf.math.log(mu[mb-1])

    ic = mu + tf.reshape(alpha[:, (mb - 1)], (dim, 1))
    # j=1

    for tc, mc in timestamps[2:]:
        # First we estimate the compensator
        inside_log = (mu - tf.math.minimum(ic, 0))/mu
        # Restart time
        # t_star = tb + torch.log(inside_log)/beta
        aux = inside_log*np.array(inside_log != 0) + tf.ones((dim,1))*np.array(inside_log == 0)

        print(((tb + tf.math.log(inside_log)/beta) < tc)*(mu*(tc-tb - tf.math.log(inside_log)/beta)))
        compensator += ((tb + tf.math.log(inside_log)/beta) < tc)*(mu*(tc-tb - tf.math.log(inside_log)/beta)) + ((ic-mu)/beta)*(aux - tf.math.exp(-beta*(tc-tb)))

        # Then, estimation of intensity before next jump.
        if mc > 0:
            ic = mu + (ic - mu)*tf.math.exp(-beta*(tc-tb))

            if ic[mc - 1] <= 0.0:
                # print("oh no", ic,mc, tc)
                # res = 1e8

                # rayon_spec = np.max(np.abs(np.linalg.eig(np.abs(alpha) / beta)[0]))
                # rayon_spec = min(rayon_spec, 0.999)
                # if 2*(tList[-1][0])/(1-rayon_spec) < 0:
                #     print("wut")
                # res = 2*(tList[-1][0])/(1-rayon_spec)

                res = 1e8+(tf.reduce_sum((mu+100)**2) + tf.reduce_sum((alpha+1)**2) + tf.reduce_sum((beta+1)**2))
                return res
            else:
                log_i = log_i**np.array([[i==(mc-1)] for i in range(dim)]) + tf.math.log(ic[mc - 1])
            # j += 1
            # print(j, ic, 1+0.45*(1-0.5**(j-1)))
            ic = ic + alpha[tf.newaxis, [mc - 1]]

        tb = tc
    likelihood = log_i - compensator
    if not(dimensional):
        likelihood = tf.reduce_sum(likelihood)
    return -likelihood
