# unit test he SVRG solver
include("../src/DMD_Util.jl")
include("../src/DMD_Type.jl")
include("../src/DMD_Func.jl")
include("../src/DMD_PG.jl")

# Generate DMD Synthetic Data
#------------------------------------------------------------------------------
# dimensions
m = 100;    # temporal dimension
n = 100;    # spatial dimension
k = 3;      # number of modes
T = Float64;
sigman = 1e-4;
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

itm = 200;
tol = 1e-10;
ptf = 10;
opts = DMD_PG_Options(itm=itm, tol=tol, ptf=ptf);

obj_his, err_his = solveDMD_withPG(params, opts);

println(besterrperm(params.a,at))
println(besterrperm(params.a,a0))
#println(params.a)
#println(at)

println(aBFunc(params)/lossFunc(X))

params2 = DMDParams(k,Xdat,t,lossFunc,lossGrad);
copyto!(params2.a,at)
println(aBFunc(params2)/lossFunc(X))
copyto!(params2.a,a0)
println(aBFunc(params2)/lossFunc(X))

