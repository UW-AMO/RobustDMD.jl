
# Robust DMD in julia

This repository contains an implementation of a robust
version of the dynamic mode decomposition (DMD) in 
julia. 

## Install

You can install this package by cloning the Git
repository:

> Pkg.add("https://github.com/UW-AMO/RobustDMD.jl.git")

or a particular version by modifying the url.

## Requirements

We recommend julia 1.5 or later.


## Changelog

- Update January 14, 2021. Overhaul of code
-- More clean up of packaging
-- Re-implementation of certain solvers and trimming
-- Fixed several efficiency issues with L2-type penalties
- Update September 27, 2019
-- clean up packaging (thanks, Maarten Pronk)
-- move examples to example folder, delete broken examples
- Update June 30, 2019
-- Merge in julia-one branch, which should still
be neutral to data type and is otherwise simpler than
the original.
- Update August 13, 2018
-- Minor change to interface of DMDParams
-- Complex{Float32} is now supported in addition to
Complex{Float64}. Eventually, all floating point types
should be acceptable but that code will not be as efficient
(the single and double precision versions are heavily
BLAS dependent).
-- Stylistic changes to be more julian

## Paper examples

The paper-examples directory includes files for generating the
figures in the paper "Robust and scalable methods for the
dynamic mode decomposition"

WARNING: the "hidden dynamics" example takes several hours
to run. This example was designed to demonstrate the
effectiveness of different penalties. Thus, a dumb
and brute force optimization strategy was employed
(start with an initialization from exact DMD and
run thousands of steps of gradient descent for each
example). In practice, the SVRG algorithm would usually
outperform this approach significantly but the appropriate
algorithm parameters can be data dependent.

### running the examples

For each example, there are two files to run

- example_example_name_dr.jl which runs the data generation step and saves the data to a sub-folder called "results" within the paper-examples folder
- plot_example_name.jl which makes a basic version of the figure after the driver has been run. The images are output to the sub-folder called "figures" within the paper-examples folder

To run the examples, you wil have to first install
the RobustDMD package and its dependencies. Then, these
files can simply be included to run them (or run from
the command line).

If you are having difficulty reproducing the
figures with the current version of the codes and
your set-up, a Manifest file which contains the
git commits of every piece of relevant code used
to generate the actual figure (with julia version 1.5.3),
is included in the top-level directory.
To use this manifest file, first copy the file
Manifest.toml.papersave to Manifest.toml.
Then, start julia. Switch to the pkg mode
by typing "]". Then, you can activate the package
and install the exact versions of the packages
used to generate the figures with

```julia
(@v1.5) pkg> activate .
(RobustDMD) pkg> instantiate
```
You can then run the examples as described above.

## TODO

- Re-implement proximal gradient descent and trimming
outer solvers (lost in julia one re-write)
- Write a more robust inner solver (black box BFGS
is inappropriate for Huber norm)
- Make proper unit tests for existing solvers (SVRG
and BFGS)


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
The type DMDParams specifies these choices.
See the documentation of DMDParams for specifics.
Additionally, a prox operation may be specified
for the outer problem in certain solvers (SVRG
for now). We recommend at least
using a prox which forces the real part of alpha
to be less than some upper bound (for numerical
stability) but any type of prox operation
may be added.

Currently implemented loss functions:
- l2 (Frobenius)
- huber

Currently implemented outer solvers (* allows a prox
operation to be specified):
- BFGS
- Stochastic Variance Reduced Gradient (SVRG) 
descent (for larger problems) *

Currently implemented inner solvers:
- Optim-based BFGS solver 


## Examples

See the examples folder for some examples of usage.
This is work in progress.

## License

The files in the "src", "test", and "examples" directories are
available under the MIT license.

The MIT License (MIT)

Copyright (c) 2018 Travis Askham, Peng Zheng, Aleksandr Aravkin, J. Nathan Kutz

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.