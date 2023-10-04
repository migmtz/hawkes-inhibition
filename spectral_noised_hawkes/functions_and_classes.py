import numpy as np
from scipy.integrate import quad
from class_and_func.hawkes_process import exp_thinning_hawkes


def bartlett_periodogram(w, tList):
    T = tList[-1]
    t_aux = np.array(tList[1:-1])
    dt = np.sum(np.exp(- 2j * np.pi * w * t_aux))
    return ((1 / T) * dt * np.conj(dt)).real


def debiaised_bartlett_periodogram(w, tList, avg_intensity):
    T = tList[-1]
    t_aux = np.array(tList[1:-1])
    dt = np.sum(np.exp(- 2j * np.pi * w * t_aux))
    dt -= avg_intensity * np.sqrt(T) * np.exp(- 1j * np.pi * w * T) * np.sinc(w * T)
    return ((1 / T) * dt * np.conj(dt)).real


def spectral_log_likelihood(theta, f, M, tList):
    T = tList[-1]
    f_array = np.array([f(j / T, theta) for j in range(1, M+1)])
    periodogram = np.array([bartlett_periodogram(j / T, tList) for j in range(1, M+1)])
    #periodogram = np.convolve((1/25)*np.ones(25), periodogram)[24:]
    if f_array[0] < 0:
        print(f, f_array[0], theta)
    #pll = -(1/T) * np.sum(np.log(f_array) + (1/f_array - 1) * periodogram)
    pll = -(1/T) * np.sum(np.log(f_array) + (1/f_array) * periodogram)

    return -pll


def integral_spectral_log_likelihood(theta, f, tList, upper_bound):
    function = lambda w: bartlett_periodogram(w, tList)/f(w, theta) + np.log(f(w, theta))

    pll = -quad(function, 0, upper_bound)[0]

    return -pll


def spectral_f_exp(w, theta):
    mu, alpha, beta = theta
    avg = mu / (1 - alpha)

    return avg * (1 + alpha * (beta**2) * (2 - alpha)/((beta*(1 - alpha))**2 + (2 * np.pi * w)**2))


def spectral_f_exp_noised(w, theta):
    mu, alpha, beta, lambda0 = theta
    f_val = spectral_f_exp(w, (mu, alpha, beta))

    return f_val + lambda0


def spectral_f_exp_grad(w, theta):
    mu, alpha, beta = theta
    avg = mu / (1 - alpha)
    D_ab = (beta * (1 - alpha)) ** 2 + (2 * np.pi * w)**2
    C_ab = 1 + alpha * (beta**2) * (2 - alpha)/D_ab
    f_val = avg * C_ab
    grad = np.zeros(3)
    grad[0] = C_ab / (1 - alpha)
    grad[1] = mu * C_ab * (2 * (beta**2) / D_ab + 1 / (1 - alpha)**2)
    grad[2] = 2 * avg * alpha * beta * (2 - alpha) * ((2 * np.pi * w)**2) / (D_ab**2)

    return f_val, grad[0], grad[1], grad[2]


def spectral_f_exp_noised_grad(w, theta):
    mu, alpha, beta, noise = theta
    f_val, grad0, grad1, grad2 = spectral_f_exp_grad(w, (mu, alpha, beta))

    return f_val + noise, grad0, grad1, grad2, 1


def spectral_log_likelihood_grad(theta, f, M, tList):
    T = tList[-1]
    f_array = np.array([f(j / T, theta) for j in range(1, M+1)])
    f_val, grad = f_array[:, 0], f_array[:, 1:]
    periodogram = np.array([bartlett_periodogram(j / T, tList) for j in range(1, M+1)])
    #pll = -(1/T) * np.sum(np.log(f_array) + (1/f_array - 1) * periodogram)
    pll = -(1/T) * np.sum(np.log(f_val) + (1/f_val) * periodogram)
    aux = (1/f_val) * (1 - (1/f_val) * periodogram)
    pll_grad = -(1/T) * np.sum(grad * aux.reshape(M, 1), axis=0)
    print(-pll)

    return -pll, -pll_grad


# def spectral_f_exp_noised(w, theta):
#     mu, alpha, beta, noise
#     return noise + (average_intensity/(2*np.pi)) * (1 + (alpha*(2*beta - alpha))/((beta-alpha)**2 + x**2))


if __name__ == "__main__":
    np.random.seed(2)
    mu = 1
    alpha = 0.3
    beta = 0.5

    noise = 2.0

    max_time = 2000.0
    burn_in = -1000

    M = 10000

    hp = exp_thinning_hawkes(mu, alpha, beta, t=burn_in, max_time=max_time)
    hp.simulate()

    times_hp = [0.0] + [t for t in hp.timestamps if t > 0] + [max_time]

    print(spectral_log_likelihood_grad((mu, alpha, beta), spectral_f_exp_grad, 2, times_hp))