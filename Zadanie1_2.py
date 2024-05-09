from Dirichlett import Dirichlett
from numpy import cos, sin, pi
import numpy as np
def main():
    label    = r"$y''=12x$"
    alpha    = lambda x: 1                                                                             # Współczynnik przy y''
    beta     = lambda x: 0                                                                             # Współczynnik przy y' 
    gamma    = lambda x: 0                                                                            # Współczynnik przy y
    f        = lambda x: 12*x                                                                     # Wyraz wolny
    solution = lambda x: 2*x**3-x
    n = 10000                                                                                          # Liczba węzłów
    a = [0, 0]                                                                                  # Argument funkcji, wartość w punkcie
    b = [1, 1]                                                                            # -''- dla prawego punktu
    dirichlet = Dirichlett(alpha, beta, gamma, f)
    dirichlet.setN(n)
    dirichlet.setAB(a, b)
    dirichlet.setLabel(label)
    dirichlet.addSolution(solution)                                    # Rozwiązanie analityczne
    lista = np.array([])
    for n in [3,10,50,100,500,1000]:
        dirichlet.setN(n)
        lista = np.append(lista, dirichlet.getError())
    print(lista)
if __name__ == '__main__':
    main()