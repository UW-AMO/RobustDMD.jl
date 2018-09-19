# unit test the set up

include("DMDType.jl")
include("DMDUtil.jl")

#--------------------------------------------------------------------
# Generate DMD Synthetic Data
#--------------------------------------------------------------------
# dimensions
m = 100;    # temporal dimension
n = 100;    # spatial dimension
k = 3;      # number of modes
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
# Create object
#--------------------------------------------------------------------
params = DMDParams(k, X, t);
vars = DMDVars(params);
svars = DMDSVars(params);