# Paper examples

This directory includes files for generating the
figures in the paper "Robust and scalable methods for the
dynamic mode decomposition"

Warning: the "hidden dynamics" example takes several hours
to run. This example was designed to demonstrate the
effectiveness of different penalties. Thus, a dumb
and brute force optimization strategy was employed
(start with a decent initialization from exact DMD and
run thousands of steps of gradient descent for each
example). The issue is that the standard L2 penalty is
indeed an inappropriate penalty for these examples and
the algorithms struggle to converge. Observe that the
penalty we advocate, the Huber norm, converges well using
BFGS using much fewer iterations.

## running the examples

For each example, there are two files to run

- example_example_name_dr.jl which runs the data generation step and saves the data to a sub-folder called "results" within the paper-examples folder
- plot_example_name.jl which makes a basic version of the figure after the driver has been run. The images are output to the sub-folder called "figures" within the paper-examples folder

To run the examples, you will likely have to first install
the RobustDMD package and its dependencies. Then, these
files can simply be included to run (or run from the command
line). 