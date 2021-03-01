using RobustDMD, Random, Printf

# Generate DMD Synthetic Data
#------------------------------------------------------------------------------
# dimensions
m = 500;    # temporal dimension
n = 1000;    # spatial dimension
k = 3;      # number of modes
T = Float64;
sigman = 1e-3;
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
# create initial guess using exact dmd
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
res = solveDMD_withOptimBFGS(params);


@printf "%7.4e relative residual after BFGS\n" sqrt(dmdobjective(params)/lossFunc(X))

params2 = DMDParams(k,Xdat,t,lossFunc,lossGrad);
copyto!(params2.a,at)
@printf "%7.4e relative residual with true evals\n" sqrt(dmdobjective(params2)/lossFunc(X))
copyto!(params2.a,a0)
@printf "%7.4e relative residual with initial guess\n" sqrt(dmdobjective(params2)/lossFunc(X))



@printf "%7.4e abs err evals after BFGS\n" besterrperm(params.a,at)
@printf "%7.4e abs err evals initial guess\n" besterrperm(a0,at)
