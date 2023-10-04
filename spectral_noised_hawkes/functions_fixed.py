import numpy as np


def spectral_f_exp_grad(w, theta):
    mu, alpha, beta = theta
    avg = mu / (1 - alpha)
    D_ab = (beta * (1 - alpha)) ** 2 + w**2
    C_ab = 1 + alpha * (beta**2) * (2 - alpha)/D_ab
    f_val = avg * C_ab / (2 * np.pi)
    grad = np.zeros(3)
    grad[0] = C_ab / (1 - alpha)
    grad[1] = mu * C_ab * (2 * (beta**2) / D_ab + 1 / (1 - alpha)**2)
    grad[2] = 2 * avg * alpha * beta * (2 - alpha) * (w**2) / (D_ab**2)
    grad /= (2 * np.pi)

    return f_val, grad[0], grad[1], grad[2]


def bartlett_periodogram(w, tList):
    T = tList[-1]
    t_aux = np.array(tList[1:-1])
    dt = np.sum(np.exp(1j * w * t_aux))
    return ((1 / (2*np.pi * T)) * dt * np.conj(dt)).real


def spectral_f_exp_noise_grad(w, theta, noise):
    mu, alpha, beta = theta
    f_val, grad0, grad1, grad2 = spectral_f_exp_grad(w, (mu, alpha, beta))

    return f_val + noise/(2 * np.pi), grad0, grad1, grad2


def spectral_ll_noise_grad_precomputed(theta, M, tList, periodogram, noise):
    T = tList[-1]
    f_array = np.array([spectral_f_exp_noise_grad(2 * np.pi * j / T, theta, noise) for j in range(1, M+1)])
    f_val, grad = f_array[:, 0], f_array[:, 1:]
    periodogram = periodogram
    #pll = -(1/T) * np.sum(np.log(f_array) + (1/f_array - 1) * periodogram)
    #print((np.log(f_val) + (1/f_val) * periodogram)[0:5])
    pll = -(1/T) * np.sum(np.log(f_val) + (1/f_val) * periodogram)
    aux = (1/f_val) * (1 - (1/f_val) * periodogram)
    pll_grad = -(1/T) * np.sum(grad * aux.reshape(M, 1), axis=0)
    return -pll, -pll_grad


def spectral_f_exp_mu_grad(w, theta, mu):
    alpha, beta, noise = theta
    f_val, grad0, grad1, grad2 = spectral_f_exp_grad(w, (mu, alpha, beta))

    return f_val + noise/(2 * np.pi), grad1, grad2, 1/(2 * np.pi)


def spectral_ll_mu_grad_precomputed(theta, M, tList, periodogram, mu):
    T = tList[-1]
    f_array = np.array([spectral_f_exp_mu_grad(2 * np.pi * j / T, theta, mu) for j in range(1, M+1)])
    f_val, grad = f_array[:, 0], f_array[:, 1:]
    periodogram = periodogram
    #pll = -(1/T) * np.sum(np.log(f_array) + (1/f_array - 1) * periodogram)
    #print((np.log(f_val) + (1/f_val) * periodogram)[0:5])
    pll = -(1/T) * np.sum(np.log(f_val) + (1/f_val) * periodogram)
    aux = (1/f_val) * (1 - (1/f_val) * periodogram)
    pll_grad = -(1/T) * np.sum(grad * aux.reshape(M, 1), axis=0)
    return -pll, -pll_grad


def spectral_f_exp_alpha_grad(w, theta, alpha):
    mu, beta, noise = theta
    f_val, grad0, grad1, grad2 = spectral_f_exp_grad(w, (mu, alpha, beta))

    return f_val + noise/(2 * np.pi), grad0, grad2, 1/(2 * np.pi)


def spectral_ll_alpha_grad_precomputed(theta, M, tList, periodogram, alpha):
    T = tList[-1]
    f_array = np.array([spectral_f_exp_alpha_grad(2 * np.pi * j / T, theta, alpha) for j in range(1, M+1)])
    f_val, grad = f_array[:, 0], f_array[:, 1:]
    periodogram = periodogram
    #pll = -(1/T) * np.sum(np.log(f_array) + (1/f_array - 1) * periodogram)
    #print((np.log(f_val) + (1/f_val) * periodogram)[0:5])
    pll = -(1/T) * np.sum(np.log(f_val) + (1/f_val) * periodogram)
    aux = (1/f_val) * (1 - (1/f_val) * periodogram)
    pll_grad = -(1/T) * np.sum(grad * aux.reshape(M, 1), axis=0)
    return -pll, -pll_grad


def spectral_f_exp_beta_grad(w, theta, beta):
    mu, alpha, noise = theta
    f_val, grad0, grad1, grad2 = spectral_f_exp_grad(w, (mu, alpha, beta))

    return f_val + noise/(2 * np.pi), grad0, grad1, 1/(2 * np.pi)


def spectral_ll_beta_grad_precomputed(theta, M, tList, periodogram, beta):
    T = tList[-1]
    f_array = np.array([spectral_f_exp_beta_grad(2 * np.pi * j / T, theta, beta) for j in range(1, M+1)])
    f_val, grad = f_array[:, 0], f_array[:, 1:]
    periodogram = periodogram
    #pll = -(1/T) * np.sum(np.log(f_array) + (1/f_array - 1) * periodogram)
    #print((np.log(f_val) + (1/f_val) * periodogram)[0:5])
    pll = -(1/T) * np.sum(np.log(f_val) + (1/f_val) * periodogram)
    aux = (1/f_val) * (1 - (1/f_val) * periodogram)
    pll_grad = -(1/T) * np.sum(grad * aux.reshape(M, 1), axis=0)
    return -pll, -pll_grad
