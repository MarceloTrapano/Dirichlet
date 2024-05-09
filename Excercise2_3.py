from Dirichlett import Dirichlett
from numpy import cos, sin, pi
import numpy as np
def main():
    label = r"$-x^2y''-2xy'+2y = -4x^2$"
    alpha =     lambda x: -x**2                                                                             # Współczynnik przy y''
    beta =      lambda x: -2*x                                                                             # Współczynnik przy y' 
    gamma =     lambda x: 2                                                                            # Współczynnik przy y
    f =         lambda x: -4*x**2                                                                     # Wyraz wolny
    solution =  lambda x: x**2 - x
    n = 10000                                                                                          # Liczba węzłów
    a = [0, 0]                                                                                  # Argument funkcji, wartość w punkcie
    b = [1,0]                                                                            # -''- dla prawego punktu
    dirichlet = Dirichlett(alpha, beta, gamma, f, True)
    dirichlet.setN(n)
    dirichlet.setAB(a, b)
    dirichlet.setLabel(label)
    dirichlet.addSolution(solution)                                    # Rozwiązanie analityczne
    lista = np.array([])
    for n in [3,10,50]:
        dirichlet.setN(n)
        lista = np.append(lista, dirichlet.getError())
    print(lista)
if __name__ == '__main__':
    main()