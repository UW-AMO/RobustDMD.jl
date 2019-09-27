# unit test he SVRG solver
include("../src/DMD_Util.jl")
include("../src/DMD_Type.jl")
include("../src/DMD_Func.jl")
include("../src/DMD_SVRG.jl")
include("../src/DMD_BFGS.jl")
include("../src/DMD_Loss.jl")

# Generate DMD Synthetic Data
#------------------------------------------------------------------------------
# dimensions
m = 500;    # temporal dimension
n = 100;    # spatial dimension
k = 3;      # number of modes
T = Float64;
# generate data
X, t, at, Bt = simDMD(m, n, k, T; seed=123456);

# additive noise
sigma = 1.0e-4 # background noise level
lds = 0.1 # larger deviation size
ldf = 0.05 # frequency of larger deviations
X1 = X .+ randn(real(eltype(X)),size(X))*sigma .+ (rand(real(eltype(X)),size(X)) .< ldf)*lds
X2 = copy(X1)
# specify the loss
kappa = lds/5.0
lossFunc1(r) = huber_func(r,kappa)
lossGrad1(r) = huber_grad!(r,kappa)
lossFunc2(r) = l2_func(r)
lossGrad2(r) = l2_grad!(r)

#
# Create object
#------------------------------------------------------------------------------
params1 = DMDParams(k, X1, t, lossFunc1, lossGrad1);
params2 = DMDParams(k, X2, t, lossFunc2, lossGrad2);
params2B = DMDParams(k, X2, t, lossFunc2, lossGrad2);
Random.randn!(params1.ar);
copyto!(params2.ar,params1.ar);

#
# Apply Solver
#------------------------------------------------------------------------------
tau = 10;
eta = 1e1;
itm = 2000;
tol = 1e-6;
ptf = 1000;

function prox_stab(ar)
    # project for numerical stability
    k = length(ar) >> 1;
    @inbounds for i = 1:k
        ar[i<<1-1] = min(0.0, ar[i<<1-1]);
    end
end


opts1 = DMD_SVRG_Options(tau, eta, params1, itm=itm, tol=tol, ptf=ptf,
                         prox=prox_stab);
opts2 = DMD_SVRG_Options(tau, eta, params2, itm=itm, tol=tol, ptf=ptf,
                         prox=prox_stab);

sigma   = 1e-3;
itm = 1000;
tol = 1e-6;
ptf = 10;
opts2B = DMD_BFGS_Options(sigma, params2B, itm=itm, tol=tol, ptf=ptf);


@time obj_his1, err_his1 = solveDMD_withSVRG(params1, opts1);
@time obj_his2, err_his2 = solveDMD_withSVRG(params2, opts2);

@time obj_his2B, err_his2B = solveDMD_withBFGS(params2B, opts2B);


@printf "%7.4e abs err evals w/ Huber\n" besterrperm(params1.a,at)
@printf "%7.4e abs err evals w/ L2\n" besterrperm(params2.a,at)
@printf "%7.4e abs err evals w/ L2 (BFGS)\n" besterrperm(params2.a,at)
