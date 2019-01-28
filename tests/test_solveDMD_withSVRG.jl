# unit test he SVRG solver
include("../src/DMD_Util.jl")
include("../src/DMD_Type.jl")
include("../src/DMD_Func.jl")
include("../src/DMD_SVRG.jl")

# Generate DMD Synthetic Data
#------------------------------------------------------------------------------
# dimensions
m = 1000;    # temporal dimension
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
τ = 10;
η = 1e0;
itm = 2000;
tol = 1e-6;
ptf = 100;

function prox_stab(ar)
    # project for numerical stability
    k = length(ar) >> 1;
    @inbounds for i = 1:k
        ar[i<<1-1] = min(0.0, ar[i<<1-1]);
    end
end


opts = DMD_SVRG_Options(τ, η, params, itm=itm, tol=tol, ptf=ptf,
                        prox=prox_stab);

obj_his, err_his = solveDMD_withSVRG(params, opts);

println(besterrperm(params.a,at))
