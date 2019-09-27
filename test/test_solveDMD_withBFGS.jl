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
k = 8;      # number of modes
T = Float64;
sigman = 1e-2;
# generate data
X, t, at, Bt = simDMD(m, n, k, T; seed=123456);
Xdat = X + sigman*Random.randn!(copy(X))
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

#
# Initial Guess
#------------------------------------------------------------------------------
# create initial guess using trap. rule with exact dmd
 a0, B0 = dmdexactestimate(m, n, k, Xdat, t);
#
# Create Object
#------------------------------------------------------------------------------
params = DMDParams(k, Xdat, t, lossFunc, lossGrad);
 copyto!(params.a, a0);
# copyto!(params.B, B0);
#Random.randn!(params.ar);
#
# Apply Solver
#------------------------------------------------------------------------------
sigma   = 1e0;
itm = 1000;
tol = 1e-10;
ptf = 1;
opts = DMD_BFGS_Options(sigma, params, itm=itm, tol=tol, ptf=ptf);

obj_his, err_his = solveDMD_withBFGS(params, opts);

println(besterrperm(params.a,at))

println(aBFunc(params)/lossFunc(X))

params2 = DMDParams(k,Xdat,t,lossFunc,lossGrad);
copyto!(params2.a,at)
println(aBFunc(params2)/lossFunc(X))
copyto!(params2.a,a0)
println(aBFunc(params2)/lossFunc(X))

