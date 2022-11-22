from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def distance_xdx(Ru, Rx, R, x, u):
    if Ru > R:
        return np.nan
    else:
        aux = Ru**2 + Rx**2
        result1 = aux - 2*(u[0]*x[0] + u[1]*x[1])
        result2 = (aux - R**2)/(2*Rx)
        result3 = np.sqrt(((aux - R**2)**2)/(4*Rx**2) - (Ru**2 - R**2))



        return result1/(result1-result2+result3)

def coord_to_polar(x, y):
    Ru = np.sqrt(x**2 + y**2)
    return Ru


if __name__ == "__main__":
    many = 10000

    pointx = [0.6, 0.5]
    Rx = coord_to_polar(pointx[0], pointx[1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(-1, 1, many)
    y = np.linspace(-1, 1, many)

    X, Y = np.meshgrid(x, y)

    r = coord_to_polar(X, Y)

    influence = np.array([[distance_xdx(r[i, j], Rx, 1, pointx, (X[i, j], Y[i, j])) for j in range(many)] for i in range(many)])

    ax.plot_surface(X, Y, influence, cmap=cm.coolwarm)

    ax.scatter(pointx[0], pointx[1], 0)

    plt.show()

