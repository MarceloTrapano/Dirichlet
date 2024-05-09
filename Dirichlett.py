import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from ThomasSolve import ThomasSolve
import time

class Dirichlett:
    '''
    Solves second order differencial equations with Dirichlett boundary conditions. 
    '''
    def __init__(self, alpha, beta, gamma, f, solveType : bool=False) -> None:
        '''
        Initiation of problem given by equation: alpha(x)*y''+beta(x)*y'+gamma(x)*y = f(x).
        
        Args:
            alpha (lambda function): alpha coefficient
            beta (lambda function): beta coefficient
            gamma (lambda function): gamma coefficient
            f (lambda function): function f
            solveType (bool, optional): if True, solves the problem using ThomasSolve. Defaults to False.'''
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.f = f
        self.solveType : bool = solveType

        self.solution = None
        self.label : str = ""
        
        self.sol_exist : bool = False
        self.set_n : bool = False
        self.set_ab : bool = False
        
        self.n : int = 1
        self.a : float = 0
        self.b : float = 2
        self.val_a : float = 0
        self.val_b : float = 0
        self.h : float = 0

        self.x : np.ndarray
        self.unknown_x : np.ndarray

    def setN(self, n : int) -> None:
        '''
        Sets the number of points.
        
        Args:
            n (int): number of points'''
        self.n = n
        self.set_n = True
        self.setH()
 
    def setLabel(self, label : str) -> None:
        '''
        Sets given equation in LaTeX to write it on plot.
        
        Args:
            label (str): label of graph'''
        self.label = label

    def setAB(self, a : list, b : list) -> None:
        '''
        Sets boundary conditions.
        
        Args:
            a (list): Two element list containing
            value of left boundary argument and value.
            b (list): Two element list containing
            value of right boundary argument and value.'''
        self.a = a[0]
        self.b = b[0]
        self.val_a = a[1]
        self.val_b = b[1]
        self.set_ab = True
        self.setH()

    def setH(self) -> None:
        '''Sets distance between points'''
        self.h = (self.b - self.a)/(self.n+1)
        self.x = np.linspace(self.a, self.b, self.n+2)
        self.unknown_x = self.x[1:self.n+1]

    def addSolution(self, solution) -> None:
        '''Add solution to the problem'''
        self.solution = solution
        self.sol_exist = True

    def solve(self) -> float:
        '''
        Solves the problem.
        
        Returns:
            float: solution depending on solve type variable.'''
        if not all([self.set_ab, self.set_n]):
            raise ValueError("Cannot solve equation without setting parameters")
        v1 : np.ndarray = np.array([])
        v2 : np.ndarray = np.array([])
        v3 : np.ndarray = np.array([])

        for i in range(len(self.unknown_x)):
            v1 = np.append(v1, -2*self.alpha(self.unknown_x[i]) + self.gamma(self.unknown_x[i])*self.h**2)
            v2 = np.append(v2, self.alpha(self.unknown_x[i]) + self.beta(self.unknown_x[i])*self.h/2)
            v3 = np.append(v3, self.alpha(self.unknown_x[i]) - self.beta(self.unknown_x[i])*self.h/2)

        
        v2 = v2[0:len(v2)-1]
        v3 = v3[1:len(v3)]

        F = list(map(lambda x: x*self.h**2, list(map(self.f, self.unknown_x))))
        G = np.zeros(len(self.unknown_x))

        G[0] = self.val_a*(self.alpha(self.unknown_x[0])-(1/2)*self.beta(self.unknown_x[0])*self.h)
        G[-1] = self.val_b*(self.alpha(self.unknown_x[-1])+(1/2)*self.beta(self.unknown_x[-1])*self.h)
        if self.solveType:
            st = time.time()
            sol = ThomasSolve(v3,v1,v2,F-G)
            et = time.time()
        else:
            A = np.diag(v1) + np.diag(v2,1) + np.diag(v3,-1)
            st = time.time()
            sol =  LA.solve(A,F-G)
            et = time.time()
        print(f"Estymowany czas działania funkcji: {et - st} sekund")
        return sol
    def plt_config(self) -> None:
        '''
        Configures plot.
        '''
        A = 6
        plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{polski}')
        plt.rc('font', family='serif')
    def getError(self) -> float:
        if not self.sol_exist:
            raise ValueError("Cannot get error without solution")
        unknown_sol = self.solve()
        approx_sol = np.insert(unknown_sol, 0, self.val_a)
        approx_sol = np.insert(approx_sol, len(approx_sol), self.val_b)
        real_sol = list(map(self.solution, self.x))
        error_sol = np.abs(approx_sol-real_sol)
        error_norm = LA.norm(error_sol, np.inf)
        return error_norm
    def show(self) -> str:
        '''
        Plots the solution.
        
        Returns:
            str: Error norm if can.'''
        self.plt_config()
        unknown_sol = self.solve()
        approx_sol = np.insert(unknown_sol, 0, self.val_a)
        approx_sol = np.insert(approx_sol, len(approx_sol), self.val_b)
        if self.sol_exist:
            real_sol = list(map(self.solution, self.x))
            error_sol = np.abs(approx_sol-real_sol)
            error_norm = LA.norm(error_sol, np.inf)
            plt.subplot(211)
            plt.grid(True)
            domain = np.linspace(self.a, self.b, 1000)
            plt.plot(domain ,list(map(self.solution, domain)), color='b', label=r"Rzeczywista funkcja")
            if self.n < 30:
                plt.scatter(self.x ,approx_sol, color='g', label=r"Aproksymacja")
            else:
                plt.plot(self.x ,approx_sol, color='g', label=r"Aproksymacja")
            plt.title(self.label)
            plt.ylabel(r"Wartości funkcji $f(x)$")
            plt.legend()
            plt.subplot(212)
            plt.grid(True)
            plt.plot(self.x ,error_sol, color='r', label=r"Błąd")
            plt.title(r"Wykres błędu")
            plt.xlabel(r"Wartości $x$")
            plt.ylabel(r"Wartość błędu")
            plt.subplots_adjust(hspace=0.4)
            plt.show()
            self.result = approx_sol
            return error_norm
        plt.grid(True)
        plt.plot(self.x ,approx_sol, color='g', label=r"Aproksymacja")
        plt.title(self.label)
        plt.xlabel(r"Wartości $x$")
        plt.ylabel(r"Wartości funkcji $f(x)$")
        plt.subplots_adjust(hspace=0.4)
        plt.show()
        self.result = approx_sol
        return ""
    
        
    def __str__(self) -> str:
        '''
        String representation of result. Shows result in plot with simple print function.
        '''
        solution = self.show()
        if self.sol_exist:
            return f"Rozwiazanie przyblizone:\n {self.result} \n Blad: {solution}"
        return f"Rozwiazanie przyblizone:\n {self.result}"
    
def main() -> None:
    alpha = lambda x: 1
    beta =  lambda x: 0
    gamma = lambda x: -4
    f =     lambda x: -4*x
    n = 10000
    a = [0, 0]
    b = [1, 2]
    dirichlet = Dirichlett(alpha, beta, gamma, f, True)
    dirichlet.setAB(a, b)
    dirichlet.addSolution(lambda x: np.exp(2)*(np.exp(4)-1)**(-1)*(np.exp(2*x)-np.exp(-2*x))+x)
    dirichlet.setLabel(r"$y''-4y = -4x$")
    dirichlet.setN(n)
    lista = np.array([])
    for n in [5,10,50,100,500,1000,5000,10000]:
        dirichlet.setN(n)
        lista = np.append(lista, dirichlet.getError())
    print(lista)

if __name__ == '__main__':
    main()