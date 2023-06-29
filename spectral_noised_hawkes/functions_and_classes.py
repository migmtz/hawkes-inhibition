import numpy as np
from scipy.integrate import quad


def bartlett_periodogram(w, tList):
    T = tList[-1]
    t_aux = tList[1:-1]
    I_0 = (len(t_aux))
    for i in range(1, len(t_aux)):
        I_0 += 2 * np.sum([np.cos(w * (t_aux[i]-y)) for y in t_aux[0:i]])
    return I_0 / (2 * np.pi * T)


def spectral_log_likelihood(theta, f, M, tList):
    T = tList[-1]
    f_array = np.array([f(2 * np.pi * j / T, theta) for j in range(M+1)])
    periodogram = np.array([bartlett_periodogram(2 * np.pi * j / T, tList) for j in range(1, M+1)])

    pll = -(1/T) * np.sum(np.log(f_array) + (1/f_array - 1) * periodogram)

    return -pll


def integral_spectral_log_likelihood(theta, f, tList, upper_bound):
    function = lambda w: bartlett_periodogram(w, tList)/f(w, theta) + np.log(f(w, theta))

    pll = -quad(function, 0, upper_bound)[0]

    return -pll


def spectral_f_exp(w, theta):
    mu, alpha, beta = theta

    return mu/(2 * np.pi) * (1 + alpha * (2 * beta - alpha)/((beta - alpha)**2 + w**2))


def spectral_f_exp_noised(w, theta):
    mu, alpha, beta, lambda0 = theta

    return( lambda0 + mu * (1 + alpha * (2 * beta - alpha)/((beta - alpha)**2 + w**2)))/(2 * np.pi)


# def spectral_f_exp_noised(w, theta):
#     mu, alpha, beta, noise
#     return noise + (average_intensity/(2*np.pi)) * (1 + (alpha*(2*beta - alpha))/((beta-alpha)**2 + x**2))



if __name__ == "__main__":
    #print(bartlett_periodogram(1, [0,1,2,3,4,5]))
    print(bartlett_periodogram(1, [0, 1, 2, 3, 4, 5]))

    tList = [1,2,3,4]
    T = 5
    prueba = 0
    w=1
    for x in tList:
        for y in tList:
            prueba += np.exp(-1j*w*(x-y))
    print(prueba/T)

    theta = (1.0, 0.5, 1.0, 0.1)
    print(spectral_log_likelihood(theta, 10, [1,2,3,4]))
