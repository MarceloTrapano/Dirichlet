from Dirichlett import Dirichlett
from numpy import cos, sin, pi
import numpy as np
def main():
    label    = r"$y'' = -\sin{x}-4\sin{(2x)}$"
    alpha    = lambda x: 1                        # Współczynnik przy y''
    beta     = lambda x: 0                        # Współczynnik przy y' 
    gamma    = lambda x: 0                        # Współczynnik przy y
    f        = lambda x: -sin(x)-4*sin(2*x)       # Wyraz wolny
    solution = lambda x: sin(x)+sin(2*x)          # Rozwiązanie analityczne
    n = 10000                                        # Liczba węzłów
    a = [0, 0]                                    # Argument funkcji, wartość w punkcie
    b = [2*pi,0]                                  # -''- dla prawego punktu
    dirichlet = Dirichlett(alpha, beta, gamma, f, True)
    dirichlet.setN(n)
    dirichlet.setAB(a, b)
    dirichlet.setLabel(label)
    dirichlet.addSolution(solution)
    lista = np.array([])
    for n in [5,10,50,100,500,1000,5000,10000]:
        dirichlet.setN(n)
        lista = np.append(lista, dirichlet.getError())
    print(lista)

if __name__ == '__main__':
    main()