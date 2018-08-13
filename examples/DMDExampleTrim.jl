####################################################################
# In this example we demonstrate using the Trimming solver. 
#
# We test the l2 penalty (with trimming) here. As such, we use
# an example problem where the outlier noise is limited to a 
# few specific sensors (this corresponds to mode=2 in the genDMD
# routine that creates the synthetic data)
#
# We also use a prox operator that forces the real part of the 
# eigenvalues alpha to be less than or equal
# to one. It is generally advisable to put some bound
# in order to keep e^(Re(alpha)*tmax) from being too large,
# e.g. overflow may occur without such a choice

using RobustDMD

T = Float32

#--------------------------------------------------------------------
# Generate DMD Synthetic Data
#--------------------------------------------------------------------
println("generate synthetic data...")
# dimensions
m = 100;    # temporal dimension
n = 100;    # spatial dimension
h = 80;     # trimming level
k = 3;      # number of modes
# generate data
sigma = T(1e-4); # size of background noise
mu = T(1); # size of spikes
p = T(0.1); # frequency of spikes (as mode=2 this determines the percentage
# of columns that are corrupted)
X, Xclean, t, alphat, betat = genDMD(m, n, k, sigma, mu; seed=123456, p=p, mode=2);


#--------------------------------------------------------------------
# Initial Guess
#--------------------------------------------------------------------
println("initialize alpha and B...")
# create initial guess using trap. rule with exact dmd
alpha0, B0 = dmdexactestimate(m,n,k,X,t);

#--------------------------------------------------------------------
# Trim l2
#--------------------------------------------------------------------
println("Trim l2 loss experiment...")
# loss functions
lossf = (z) -> l2_func(z);
lossg = (z) -> l2_grad!(z);
l2Params = DMDParams(k, X, t, lossf, lossg);
l2DMD = DMDVariables(alpha0, B0, l2Params);
# wrapper: sets up inner solve specialized for l2
l2Svars = DMDVPSolverVariablesLSQ(l2Params);
# prox operator
function prox_mr(αr)
    k = length(αr) >> 1;
    for i = 1:k
        αr[i<<1-1] = min(T(1.0), αr[i<<1-1]);
    end
end

# apply solver
options = Trim_options(h,300,T(1e-5),T(1.2),true,true,10,prox=prox_mr);
Trim_solve_DMD!(l2DMD, l2Params, l2Svars, options);


#--------------------------------------------------------------------
# Compare Result
#--------------------------------------------------------------------
println("compare result...")
err0 = besterrperm(alpha0, alphat);
err1 = besterrperm(l2DMD.alpha, alphat);
println("err exact DMD ..... ", err0);
println("err w/ trimming ... ", err1);
