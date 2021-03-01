using RobustDMD, Random, Printf

# Generate DMD Synthetic Data
# ------------------------------------------------------------------------------
# dimensions
m = 500;    # temporal dimension
n = 100;    # spatial dimension
k = 3;      # number of modes
T = Float64;
sigma = 1e-3;
# generate data
X, t, at, Bt = simDMD(m, n, k, T; seed=123456);
Xdat = X + sigma * Random.randn!(copy(X))
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
# ------------------------------------------------------------------------------
params = DMDParams(k, Xdat, t, lossFunc, lossGrad);
#
# Initial Guess
# ------------------------------------------------------------------------------
# create initial guess using exact dmd
a0, B0 = dmdexactestimate(m, n, k, Xdat, t);
#
# Create Object
# ------------------------------------------------------------------------------
params = DMDParams(k, Xdat, t, lossFunc, lossGrad, inner_directl2=true);
copyto!(params.a, a0);
copyto!(params.B, B0);
# Random.randn!(params.ar);
#
# Apply Solver
# ------------------------------------------------------------------------------
tau = 10;
eta = 1e1;
dms = 500;
itm = 2000;
tol = 1e-6;
ptf = 10;

function prox_stab(ar)
    # project for numerical stability
    k = length(ar) >> 1;
    @inbounds for i = 1:k
        ar[i << 1 - 1] = min(2.0, ar[i << 1 - 1]);
    end
end


opts = DMD_SPG_Options(tau, eta, dms=dms, itm=itm, tol=tol, ptf=ptf,
                       prox=prox_stab);

@time obj_his, err_his = solveDMD_withSPG(params, opts);

@printf "%7.4e relative residual after SPG\n" sqrt(dmdobjective(params) / lossFunc(X))

params2 = DMDParams(k, Xdat, t, lossFunc, lossGrad);
copyto!(params2.a,at)
@printf "%7.4e relative residual with true evals\n" sqrt(dmdobjective(params2) / lossFunc(X))
copyto!(params2.a,a0)
@printf "%7.4e relative residual with initial guess\n" sqrt(dmdobjective(params2) / lossFunc(X))



@printf "%7.4e abs err evals after SPG\n" besterrperm(params.a, at)
@printf "%7.4e abs err evals initial guess\n" besterrperm(a0, at)