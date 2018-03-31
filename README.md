
# Robust DMD in julia

NOTE: this is a beta version and may contain bugs. The
interface may be changed significantly in future releases.

This repository contains an implementation of a robust
version of the dynamic mode decomposition (DMD) in 
julia. 

## Requirements

The examples require the Munkres package. Otherwise the 
library is self-contained.

> Pkg.add("Munkres")

## About the robust DMD

For reference, see our preprint [here](https://arxiv.org/pdf/1712.01883.pdf).

The DMD is a popular dimensionality reduction 
tool which approximates time series data by a sum of 
exponentials. Suppose that X is a matrix where the ith
row (out of m) is a sample of some n dimensional system 
at time t(i). For a specified k, the DMD solves the 
nonlinear least squares problem 

> min_{alpha,B} rho(X - F(alpha;t) B)

where F(alpha;t) is a m by k matrix given by 

> F(alpha;t)_ij = exp(alpha(j)t(i))

and B is a k by n matrix of coefficients. Conceptually,
this corresponds to a best fit linear dynamical system
approximation of the data, i.e. the data are approximated
by the solution of dx/dt = Ax where the matrix A
has k non-zero eigenvalues alpha and corresponding 
eigenvectors given by the rows of B.

The standard least squares approach would set rho
to be the Frobenius norm (the sum of the squares of
the entries). This software enables additional
types of robust penalties: the Huber penalty and 
a trimming approach (for either the Frobenius norm
or Huber penalty). Being more ad hoc, both of these
penalties require the choice of a parameter. For 
the huber penalty, the parameter kappa decides the 
transition point for a l2-type penalty to a l1-type
penalty. The huber penalty is defined to be 

> rho(x) = |x|^2/2 for |x| <= kappa
> rho(x) = kappa*|x|-kappa^2/2 for |x| > kappa ,

the idea being that it shrinks small Gaussian type
error and is unbiased by large deviations. A good
choice for kappa is to set it to an estimate
of the standard deviation of the additive Gaussian
type noise present in the system. 

A trimming penalty adaptively chooses columns of 
X (which are commonly interpreted as spatial locations 
or individual sensors) to remove from the fit. 
The number of columns to keep must be chosen in
advance. This option is particularly well suited to
a problem for which you suspect specific sensors 
to be broken, but cannot identify them in advance.

There is also an implementation of a student's-T
type penalty (very fat-tailed). Our experience 
has been mixed with this penalty and we currently
don't recommend it.

When the function rho is the Frobenius norm (the
sum of the squares of the entries), the solution
of this problem is well-understood and fast methods
are readily available (see the classic varpro literature
and a MATLAB implementation (here)[https://github.com/duqbo/optdmd]
as well as a julia implementation (beta)
(here)[https://github.com/duqbo/varpro2-jl]).

## About the software

See the examples folder for a few usage examples.

Note that the algorithm is based on the variable
projection framework: the outer solver only concerns
itself with minimizing the objective

> f_2(alpha) = min_B  rho(X - F(alpha;t) B)

as a function of alpha alone, so that at each step
the problem min_B rho(X-F(alpha;t)B) must be solved
for a fixed alpha (this is called the inner
solve). 

The software is self-consciously modular. To
solve a particular problem, the user (1)
specifies the fitting problem data and parameters,
(2) specifies a penalty type, (3) specifies an
inner solver, and then (4) runs the outer solver.
The types DMDParams, DMDVariables, and 
DMDVPSolverVariables specify these choices.
See the documentation of DMDParams, DMDVariables,
and DMDVPSolverVariables in DMDType.jl for specifics.
Additionally, a prox operation may be specified
for the outer problem. We recommend at least
using a prox which forces the real part of alpha
to be less than some upper bound (for numerical
stability) but any type of prox operation
may be added. 

Currently implemented loss functions:
- l2 (Frobenius)
- huber
- student's T (currently not recommended)

Currently implemented outer solvers (all allow a prox
operation to be specified):
- Proximal gradient descent
- Stochastic Variance Reduced Gradient (SVRG) 
descent (for larger problems)
- Trimming gradient descent

Currently implemented inner solvers:
- BFGS with warm starts and regularization
- Closed form solution (LSQ, l2 penalty only! this
simply solves the least squares problem for B
using linear-algebraic methods)

## Examples

Each of the solvers is demonstrated in the files 
DMDExamplePG.jl (proximal gradient), DMDExampleTrim.jl
(trimming DMD), and DMDExampleSVRG.jl (SVRG solver).
We also show how to use a generic solver for the outer
problem by using the alphafunc and alphagrad! utilities
by sending these to the same BFGS solver used for the 
interior problem (see DMDExample.jl).

## License

The files in the "src" and "examples" directories are available under the MIT license.

The MIT License (MIT)

Copyright (c) 2018 Travis Askham, Peng Zheng, Aleksandr Aravkin, J. Nathan Kutz

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.