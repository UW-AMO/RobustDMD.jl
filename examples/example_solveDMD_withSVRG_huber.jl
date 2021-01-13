using RobustDMD, Printf, Random

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
sigma = 1.0e-6 # background noise level
lds = 0.1 # larger deviation size
ldf = 0.05 # frequency of larger deviations
Xdat = (X .+ randn(real(eltype(X)),size(X))*sigma
      .+ (rand(real(eltype(X)),size(X)) .< ldf)*lds)
# specify the loss
kappa = 10.0*sigma
lossFunc1(r) = huber_func(r,kappa)
lossGrad1(r) = huber_grad!(r,kappa)
lossFunc2(r) = l2_func(r)
lossGrad2(r) = l2_grad!(r)

a0, B0 = dmdexactestimate(m, n, k, Xdat, t, dmdtype="trap");


#
# Create object
#------------------------------------------------------------------------------
params1 = DMDParams(k, Xdat, t, lossFunc1, lossGrad1);
params2 = DMDParams(k, Xdat, t, lossFunc2, lossGrad2, inner_directl2=true);
copyto!(params1.a,a0)
copyto!(params2.a,a0)

#
# Apply Solver
#------------------------------------------------------------------------------
tau = 10;
eta = 2e1;
itm = 5000;
tol = 1e-7;
ptf = 100;

function prox_stab(ar)
    # project for numerical stability
    k = length(ar) >> 1;
    @inbounds for i = 1:k
        ar[i<<1-1] = min(2.0, ar[i<<1-1]);
    end
end


opts = DMD_SVRG_Options(tau, eta, itm=itm, tol=tol, ptf=ptf,
                        prox=prox_stab);

@time obj_his1, err_his1 = solveDMD_withSVRG(params1, opts);
@time obj_his2, err_his2 = solveDMD_withSVRG(params2, opts);

@printf "%7.4e abs err evals w/ Huber\n" besterrperm(params1.a,at)
@printf "%7.4e abs err evals w/ L2\n" besterrperm(params2.a,at)
