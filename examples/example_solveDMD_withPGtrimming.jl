using RobustDMD, Random, Printf, StatsBase

# Generate DMD Synthetic Data
#------------------------------------------------------------------------------
# dimensions
m = 500;    # temporal dimension
n = 100;    # spatial dimension
k = 3;      # number of modes
T = Float64;
sigman = 1e-5;
# generate data
X, t, at, Bt = simDMD(m, n, k, T; seed=123456);
# firing rate
p = 0.05; sigmabroke = 1e-1
ncol = Integer(round(n*p))
icol = sample(1:n,ncol,replace=false)
Xdat = X + sigman*Random.randn!(copy(X))
Xdat[:,icol] .+= sigmabroke*Random.randn(m,length(icol))

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
 a0, B0 = dmdexactestimate(m, n, k, Xdat, t, dmdtype="trap");
#
# Create Object
#------------------------------------------------------------------------------

# without trimming
params = DMDParams(k, Xdat, t, lossFunc, lossGrad, inner_directl2=true);
copyto!(params.a, a0);

# with trimming
paramstrim = DMDParams(k, Xdat, t, lossFunc, lossGrad, nkeep=80, inner_directl2=true);
copyto!(paramstrim.a, a0);

#
# Apply Solver
#------------------------------------------------------------------------------

itm = 400;
tol = 1e-7;
ptf = 10;
opts = DMD_PG_Options(itm=itm, tol=tol, ptf=ptf);


@printf "running proximal gradient without trimming ...\n"
obj_his, err_his = solveDMD_withPG(params, opts);
@printf "running proximal gradient with trimming\n"
obj_histrim, err_histrim = solveDMD_withPG(paramstrim, opts);


@printf "Results without trimming ...\n"
@printf "%7.4e relative residual after PG\n" sqrt(dmdobjective(params)/lossFunc(X))

params2 = DMDParams(k,Xdat,t,lossFunc,lossGrad);
copyto!(params2.a,at)
@printf "%7.4e relative residual with true evals\n" sqrt(dmdobjective(params2)/lossFunc(X))
copyto!(params2.a,a0)
@printf "%7.4e relative residual with initial guess\n" sqrt(dmdobjective(params2)/lossFunc(X))

@printf "%7.4e abs err evals after PG\n" besterrperm(params.a,at)
@printf "%7.4e abs err evals initial guess\n" besterrperm(a0,at)

@printf "... end results without trimming\n"


@printf "Results with trimming ...\n"
@printf "%7.4e relative residual after PG\n" sqrt(dmdobjective(paramstrim)/lossFunc(X))

params2 = DMDParams(k,Xdat,t,lossFunc,lossGrad);
copyto!(params2.a,at)
@printf "%7.4e relative residual with true evals\n" sqrt(dmdobjective(params2)/lossFunc(X))
copyto!(params2.a,a0)
@printf "%7.4e relative residual with initial guess\n" sqrt(dmdobjective(params2)/lossFunc(X))

@printf "%7.4e abs err evals after PG\n" besterrperm(paramstrim.a,at)
@printf "%7.4e abs err evals initial guess\n" besterrperm(a0,at)

@printf "... end results with trimming\n"
