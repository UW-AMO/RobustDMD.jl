include("BFGS.jl")
include("DMDVPSolvers.jl")
include("DMDType.jl")
include("DMDUtil.jl")
include("DMDLoss.jl")
include("DMDFunc.jl")
include("OptimizerStats.jl")

include("gradientTest.jl")

#--------------------------------------------------------------------
# Generate DMD Synthetic Data
#--------------------------------------------------------------------
# dimensions
m = 100;    # temporal dimension
n = 100;    # spatial dimension
k = 3;      # number of modes
sigma = 1e-4
mu = 100
# generate data
X, t, alphat, betat = genDMD(m, n, k, sigma, mu; seed=123456);

#
# optimization parameters
#

#--------------------------------------------------------------------
# Initial Guess
#--------------------------------------------------------------------
# create initial guess using trap. rule with exact dmd
alpha0, B0 = dmdexactestimate(m,n,k,X,t);

# #--------------------------------------------------------------------
# # l2 Trial
# #--------------------------------------------------------------------
# # DMD variables
# l2DMD = DMDVariables(alpha0, B0, params);
# # loss functions
# lossf = (z) -> l2_func(z);
# lossg = (z) -> l2_grad!(z);
# # check gradient of b[id]
# gbr = zeros(2*k);
# function funcb(br)
#     id = 50
#     bgrad_sub!(br, gbr, l2DMD, params, id, lossg);
#     f = bfunc_sub(br, l2DMD, params, id, lossf);
#     return f, copy(gbr)
# end
# gradientTest(funcb, randn(2*k), 1e-2)

# # check gradient of alpha
# galphar = zeros(2*k);
# VPsolver! = (z) -> z;
# function funcalpha(alphar)
#     alphagrad!(alphar, galphar, l2DMD, params, lossg, VPsolver!);
#     f = alphafunc(alphar, l2DMD, params, lossf, VPsolver!);
#     return f, copy(galphar)
# end
# # func = (alphar) -> alphagrad!(alphar, galphar, l2DMD, params, lossf, lossg);
# gradientTest(funcalpha, randn(2*k), 1e-2)

#--------------------------------------------------------------------
# Student's T Trial
#--------------------------------------------------------------------
# DMD variables

nu = 0.5;
# loss functions
lossf = (z) -> st_func(z, nu);
lossg = (z) -> st_grad!(z, nu);
kappa = 1.0e-3
lossf = (z) -> huber_func(z, kappa);
lossg = (z) -> huber_grad!(z, kappa);

h = n
params = DMDParams(k,X,t,lossf,lossg)
StDMD = DMDVariables(alpha0, B0, params);
StSvars = DMDVPSolverVariablesNull(params)

# check gradient of b[id]
gbr = zeros(2*k);
function funcb(br)
    id = 50
    bgrad_sub!(br, gbr, StDMD, params, id);
    f = bfunc_sub(br, StDMD, params, id);
    return f, copy(gbr)
end
gradientTest(funcb, randn(2*k), 1e-2)

# check gradient of alpha
galphar = zeros(2*k);
function funcalpha(alphar)
    alphagrad!(alphar, galphar, StDMD, params, StSvars);
    f = alphafunc(alphar, StDMD, params, StSvars);
    return f, copy(galphar)
end
# func = (alphar) -> alphagrad!(alphar, galphar, l2DMD, params, lossf, lossg);
gradientTest(funcalpha, randn(2*k), 1e-2)
