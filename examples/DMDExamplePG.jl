####################################################################
# In this example we demonstrate using the Proximal Gradient solver
# with a prox operator that enforces that the real part of the 
# eigenvalues alpha is nonpositive.
#
# We test the l2 penalty here (a more or less arbitrary choice)

using RobustDMD

T = Float32

# using PyPlot
#--------------------------------------------------------------------
# Generate DMD Synthetic Data
#--------------------------------------------------------------------
println("generate synthetic data...")
# dimensions
m = 100;    # temporal dimension
n = 100;    # spatial dimension
h = n; # h is not used by this solver (only Trim)
k = 3;      # number of modes
# generate data
sigma = T(1e-4); # size of background noise
mu = T(0.0); # size of spikes (set to zero here for l2 fitting example)
p = T(0.1); # frequency of spikes (percentage of corrupted entries)
X, Xclean, t, alphat, betat = genDMD(m, n, k, sigma, mu; seed=123456, p=p);

#--------------------------------------------------------------------
# Initial Guess
#--------------------------------------------------------------------
println("initialize alpha and B...")
# create initial guess using trap. rule with exact dmd
alpha0, B0 = dmdexactestimate(m,n,k,X,t);

#--------------------------------------------------------------------
# Proximal Gradient L2
#--------------------------------------------------------------------
println("Proximal Gradient l2 loss experiment...")
# loss functions
lossf = (z) -> l2_func(z);
lossg = (z) -> l2_grad!(z);
l2Params = DMDParams(k, X, t, lossf, lossg);
l2DMD = DMDVariables(alpha0, B0, l2Params);
# wrapper: sets up inner solve specialized for l2
l2Svars = DMDVPSolverVariablesLSQ(l2Params) 

# prox function, projects to real part <= 0.0
function prox_mr(αr)
    k = length(αr) >> 1;
    for i = 1:k
        αr[i<<1-1] = min(T(0.0), αr[i<<1-1]);
    end
end
options = PG_options(1000,T(1e-6),true,true,10,prox_mr);

# apply solver
stats = PG_solve_DMD!(l2DMD, l2Params, l2Svars, options);

# check performance
err0 = besterrperm(alpha0, alphat);
err1 = besterrperm(l2DMD.alpha, alphat);

println("err exact DMD ..... ", err0)
println("err l2 fit ........ ", err1)
