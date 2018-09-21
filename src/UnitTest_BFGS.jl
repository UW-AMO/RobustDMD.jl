# unit test he BFGS solver

include("DMDType.jl")
include("DMDUtil.jl")
include("DMDFunc.jl")
include("DMD_BFGS.jl")

#--------------------------------------------------------------------
# Generate DMD Synthetic Data
#--------------------------------------------------------------------
# dimensions
m = 100;    # temporal dimension
n = 10000;    # spatial dimension
k = 5;      # number of modes
sigma = 1e-4;
mu = 100;
# generate data
X, Xt, t, alphat, betat = genDMD(m, n, k, sigma, mu; seed=123456);

#--------------------------------------------------------------------
# Initial Guess
#--------------------------------------------------------------------
# create initial guess using trap. rule with exact dmd
alpha0, B0 = dmdexactestimate(m,n,k,X,t);

#--------------------------------------------------------------------
# Create Object
#--------------------------------------------------------------------
params = DMDParams(k, X, t);
vars = DMDVars(params);
svars = DMDSVars(params);
copy!(vars.a, alpha0);
copy!(vars.B, B0);

#--------------------------------------------------------------------
# Apply Solver
#--------------------------------------------------------------------
σ   = 1e-3;
itm = 1000;
tol = 1e-6;
ptf = 10;
opts = DMD_BFGS_Options(σ, params, itm=itm, tol=tol, ptf=ptf);

obj_his, err_his = solveDMD_withBFGS(vars, params, svars, opts);
