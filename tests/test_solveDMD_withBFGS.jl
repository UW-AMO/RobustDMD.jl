# unit test he SVRG solver
include("../src/DMD_Util.jl")
include("../src/DMD_Type.jl")
include("../src/DMD_Func.jl")
include("../src/DMD_BFGS.jl")

# Generate DMD Synthetic Data
#------------------------------------------------------------------------------
# dimensions
m = 100;    # temporal dimension
n = 100;    # spatial dimension
k = 3;      # number of modes
T = Float64;
# generate data
X, t, at, Bt = simDMD(m, n, k, T; seed=123456);
# specify the loss
function lossFunc(r)
	return sum(abs2, r)
end
#
function lossGrad(r)
	conj!(r)
end
#
# Create object
#------------------------------------------------------------------------------
params = DMDParams(k, X, t, lossFunc, lossGrad);
#
# Initial Guess
#------------------------------------------------------------------------------
# create initial guess using trap. rule with exact dmd
# a0, B0 = dmdexactestimate(m, n, k, X, t);
#
# Create Object
#------------------------------------------------------------------------------
params = DMDParams(k, X, t, lossFunc, lossGrad);
# copyto!(params.a, a0);
# copyto!(params.B, B0);
Random.randn!(params.ar);
#
# Apply Solver
#------------------------------------------------------------------------------
σ   = 1e-3;
itm = 1000;
tol = 1e-6;
ptf = 10;
opts = DMD_BFGS_Options(σ, params, itm=itm, tol=tol, ptf=ptf);

obj_his, err_his = solveDMD_withBFGS(params, opts);
