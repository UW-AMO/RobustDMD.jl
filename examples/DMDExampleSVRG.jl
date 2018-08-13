####################################################################
# In this example we demonstrate using the Stochastic Variance 
# Reduced Gradient (SVRG) descent solver. As this solver subsamples
# the gradient for each step, it is capable of doing larger
# problems more efficiently.
#
# We test the Huber penalty here
#
# We also use a prox operator that forces the real part of the 
# eigenvalues alpha to be less than or equal
# to one. It is generally advisable to put some bound
# in order to keep e^(Re(alpha)*tmax) from being too large,
# e.g. overflow may occur without such a choice

include("../src/RobustDMD.jl")
using RobustDMD

T = Float32

#--------------------------------------------------------------------
# Generate DMD Synthetic Data
#--------------------------------------------------------------------
println("generate synthetic data...")

srand(8675309)

# dimensions
m = 100;    # temporal dimension
n = 1000;    # spatial dimension
k = 3;      # number of modes
# generate data
sigma = T(1e-4); # size of background noise
mu = T(1); # size of spikes
p = T(0.1); # frequency of spikes (percentage of corrupted entries)
xdat, xclean, t, alphat, betat = genDMD(m, n, k, sigma, mu; seed=123456, p=p);

#--------------------------------------------------------------------
# Initial Guess
#--------------------------------------------------------------------
println("initialize alpha and B...")
# create initial guess using trap. rule with exact dmd
alpha0, B0 = dmdexactestimate(m,n,k,xdat,t);

#--------------------------------------------------------------------
# Huber Trial
#--------------------------------------------------------------------

println("Huber loss experiment...")

# loss functions
kappa = T(10*sigma);
lossf = (z) -> huber_func(z,kappa);
lossg = (z) -> huber_grad!(z,kappa);

h = n
huberParams = DMDParams(k,xdat,t,lossf,lossg)

# DMD variables
huberDMD = DMDVariables(alpha0, B0, huberParams);
# solver variables
huberSvars = DMDVPSolverVariablesBFGS(huberParams,tol=T(1e-5))

# prox function, projects to real part <= 0.0
function prox_mr(αr)
    k = length(αr) >> 1;
    for i = 1:k
        αr[i<<1-1] = min(T(1.0), αr[i<<1-1]);
    end
end

# apply solver
options = SVRG_options(2000,T(1e-7),10,T(10.0),100000,100,true,true,100,
                       prox=prox_mr);

stats= SVRG_solve_DMD!(huberDMD, huberParams, huberSvars,
                         options);
objs = stats.objs
errs = stats.errs
noi = stats.noi
objs = objs[1:noi+1]
errs = errs[2:noi+1]

noi = stats.noi
# plot convergent history
#semilogy(errs);
#savefig("err_his.eps");

#--------------------------------------------------------------------
# Compare Result
#--------------------------------------------------------------------
println("compare result...")
err0 = besterrperm(alpha0, alphat);
# err1 = besterrperm(l2DMD.alpha, alphat);
err2 = besterrperm(huberDMD.alpha, alphat);
# err3 = besterrperm(stDMD.alpha, alphat);
println("err exact DMD ..... ", err0)
# println("err 1 ", err1)
println("err huber fit ..... ", err2)
# println("err 3 ", err3)
