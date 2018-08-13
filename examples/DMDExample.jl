####################################################################
# In this example we demonstrate using a generic solver
# (our version of BFGS) by sending in the alphafunc and alphagrad!
# functions as the objective and gradient.
#
# Both l2 and huber penalties are demonstrated

#using RobustDMD
include("../src/RobustDMD.jl")
using RobustDMD

T = Float32

#--------------------------------------------------------------------
# Generate DMD Synthetic Data
#--------------------------------------------------------------------
println("generate synthetic data...")
# dimensions
m = Int32(100);    # temporal dimension
n = Int16(100);    # spatial dimension
h = 100;
k = 3;      # number of modes
# generate data
sigma = T(1e-4); # size of background noise
mu = T(0.1); # size of spikes
p = T(0.1); # frequency of spikes
X, Xclean, t, alphat, betat = genDMD(m, n, k, sigma, mu; seed=123456, p=p);


#--------------------------------------------------------------------
# Initial Guess
#--------------------------------------------------------------------
println("initialize alpha and B...")
# create initial guess using trap. rule with exact dmd
alpha0, B0 = dmdexactestimate(m,n,k,X,t);

#--------------------------------------------------------------------
# l2 Trial
#--------------------------------------------------------------------
println("l2 loss experiment...")
# DMD variables and parameters
# loss functions
lossf = (z) -> l2_func(z);
lossg = (z) -> l2_grad!(z);
l2Params = DMDParams(k, X, t, lossf, lossg);
l2DMD = DMDVariables(alpha0, B0, l2Params);
# solver variables
l2Svars = DMDVPSolverVariablesLSQ(l2Params);
# define outer problem
func = (alphar) -> alphafunc(alphar, l2DMD, l2Params, l2Svars);
grad! = (galphar, alphar) -> alphagrad!(alphar, galphar, l2DMD, l2Params, l2Svars);
# apply solver
alpha_BFGS_opts = BFGS_options(100, T(1e-6), true, true, true, 1);
alpha_BFGS_vars = BFGS_vars(2*k,T)
@time my_res = My_BFGS(func,grad!,l2DMD.alphar,alpha_BFGS_opts,alpha_BFGS_vars);

#--------------------------------------------------------------------
# Huber Trial
#--------------------------------------------------------------------
println("Huber loss experiment...")
# DMD variables
# loss functions
kappa = T(10*sigma);
lossf = (z) -> huber_func(z,kappa);
lossg = (z) -> huber_grad!(z,kappa);
huberParams = DMDParams(k, X, t, lossf, lossg);
huberDMD = DMDVariables(alpha0, B0, huberParams);
# solver variables
huberSvars = DMDVPSolverVariablesBFGS(huberParams,tol=T(1e-5));
# define outer problem
func = (alphar) -> alphafunc(alphar, huberDMD, huberParams, huberSvars);
grad! = (galphar, alphar) -> alphagrad!(alphar, galphar, huberDMD, huberParams, huberSvars);
# apply solver
alpha_BFGS_opts = BFGS_options(100, T(1e-6), true, true, true, 1);
alpha_BFGS_vars = BFGS_vars(2*k,T)
@time my_res = My_BFGS(func,grad!,huberDMD.alphar,alpha_BFGS_opts,alpha_BFGS_vars);

#--------------------------------------------------------------------
# Compare Result
#--------------------------------------------------------------------
println("compare result...")
err0 = besterrperm(alpha0, alphat);
err1 = besterrperm(l2DMD.alpha, alphat);
err2 = besterrperm(huberDMD.alpha, alphat);

println("err exact DMD ..... ", err0)
println("err l2 fit ........ ", err1)
println("err huber fit ..... ", err2)
