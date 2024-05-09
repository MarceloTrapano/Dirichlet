from Dirichlett import Dirichlett
from numpy import cos, sin, pi
import numpy as np
def main():
    label    = r"$y''-y'-2y = \cos{x}$"
    alpha    = lambda x: 1                                                                             # Współczynnik przy y''
    beta     = lambda x: -1                                                                             # Współczynnik przy y' 
    gamma    = lambda x: -2                                                                            # Współczynnik przy y
    f        = lambda x: cos(x)                                                                     # Wyraz wolny
    solution = lambda x: -(sin(x)+3*cos(x))/10
    n = 10000                                                                                          # Liczba węzłów
    a = [0, -3/10]                                                                                  # Argument funkcji, wartość w punkcie
    b = [pi/2,-1/10]                                                                            # -''- dla prawego punktu
    dirichlet = Dirichlett(alpha, beta, gamma, f, True)
    dirichlet.setN(n)
    dirichlet.setAB(a, b)
    dirichlet.setLabel(label)
    dirichlet.addSolution(solution)                                    # Rozwiązanie analityczne
    lista = np.array([])
    for n in [5,10,50,100,500,1000,5000,10000]:
        dirichlet.setN(n)
        lista = np.append(lista, dirichlet.getError())
    print(lista)
if __name__ == '__main__':
    main()