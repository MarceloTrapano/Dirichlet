import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ImageHandler import HandlerLineImage

dataset = pd.read_csv('Dirichlett.csv')

x = list(dataset['n'])


PythonSolve = list(dataset['LinAlgSolve'])
MatlabSolve = list(dataset['Linsolve'])
PythonThomas = list(dataset['ThomasSolveP'])
MatlabThomas = list(dataset['ThomasSolveM'])
A = 6
plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{polski}')
plt.rc('font', family='serif')
plt.subplot(311)
plt.grid(True)
python,  = plt.plot(x[0:9], PythonSolve[0:9], label='Python', color="b")
matlab,  = plt.plot(x[0:9], MatlabSolve[0:9], label='Matlab', color="r")
plt.ylabel('Czas w sekundach [$s$]')
plt.title('Wbudowana metoda rozwiązywania układów równań')
plt.legend(handler_map={python : HandlerLineImage("images.jpeg"), matlab : HandlerLineImage("Matlab_Logo.png")}, 
   handlelength=4, labelspacing=0.25, fontsize=10, borderpad=0.4, 
    handletextpad=0.0, borderaxespad=0.0)
plt.subplot(312)
plt.grid(True)
python,  = plt.plot(x, PythonThomas, label='Python', color="b")
matlab,  = plt.plot(x, MatlabThomas, label='Matlab', color="r")
plt.ylabel('Czas w sekundach [$s$]')
plt.title('Metoda Thomasa')
plt.legend(handler_map={python : HandlerLineImage("images.jpeg"), matlab : HandlerLineImage("Matlab_Logo.png")}, 
   handlelength=4, labelspacing=0.25, fontsize=10, borderpad=0.4, 
    handletextpad=0.0, borderaxespad=0.0)
plt.subplot(313)
plt.grid(True)
python,  = plt.plot(x[0:5], PythonThomas[0:5], label='Python', color="b")
matlab,  = plt.plot(x[0:5], MatlabThomas[0:5], label='Matlab', color="r")
plt.ylabel('Czas w sekundach [$s$]')
plt.xlabel('Wielkość problemu $n$')
plt.title('Metoda Thomasa, zbliżenie na wcześniejsze wartości')
plt.legend(handler_map={python : HandlerLineImage("images.jpeg"), matlab : HandlerLineImage("Matlab_Logo.png")}, 
   handlelength=4, labelspacing=0.25, fontsize=10, borderpad=0.4, 
    handletextpad=0.0, borderaxespad=0.0)
plt.subplots_adjust(hspace=0.4)
plt.show()