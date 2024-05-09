This repository contains class which can solve second order differential equations with Dirichlet boundary conditions. This class has methods created to implement this kind of problem. To initiate problem you need to specify equation given by a(x)y''+b(x)y'+c(x) = f(x). Arguments are given as lambda functions. There exists optional argument solveType, which can be used to choose, if you want to compute system of linear equations by Thomas method (Thomas method can be used only for tridiagonal matrices). To solve equation we need to add boundary conditions and specify number of nodes. Without them algorythm raises error when calling solving methods. We can also add real solution to problem and evaluate error. Class contains methods that can visualize solution with LaTeX expressions.

Libraries needed to properly run code:
-MatPlotLib
-Numpy
