# Simple comparison of algrithms, including PG, SPG, and SVRG
# on the efficiency of the iterations.
# Here we use synthetic data
using RobustDMD, Random, Printf, JLD, Plots

# Generate DMD Synthetic Data
# ------------------------------------------------------------------------------
# dimensions
m = 512;    # temporal dimension
n = 1000;    # spatial dimension
k = 3;      # number of modes
T = Float64;
sigma = 1e-2;
# generate data
X, t, at, Bt = simDMD(m, n, k, T; seed=123456);
Xdat = X + sigma * Random.randn!(copy(X))
# specify the loss and prox
function lossFunc(r)
	return sum(abs2, r)
end

function lossGrad(r)
	conj!(r)
end

function prox_stab(ar)
    # project for numerical stability
    k = length(ar) >> 1;
    @inbounds for i = 1:k
        ar[i << 1 - 1] = min(2.0, ar[i << 1 - 1]);
    end
end

# Create object
# ------------------------------------------------------------------------------
pg_params = DMDParams(k, Xdat, t, lossFunc, lossGrad);
spg_params = DMDParams(k, Xdat, t, lossFunc, lossGrad);
svrg_params = DMDParams(k, Xdat, t, lossFunc, lossGrad);

init_obj = dmdobjective(pg_params, updatep=false);


# Create solver options
# ------------------------------------------------------------------------------
tau = 20;
eta = 5e-4;
tol = 0.0;
true_obj = true;
pg_opts = DMD_PG_Options(itm=100, tol=tol, prox=prox_stab);
spg_opts = DMD_SPG_Options(tau, eta, dms=100, itm=2000, tol=tol,
                           true_obj=true_obj, prox=prox_stab);
svrg_opts = DMD_SVRG_Options(tau, eta, itm=2000, tol=tol, true_obj=true_obj,
                             prox=prox_stab);

# Get results
# ------------------------------------------------------------------------------
pg_obj_his, pg_err_his, pg_lns_his = solveDMD_withPG(pg_params, pg_opts);
spg_obj_his, spg_err_his = solveDMD_withSPG(spg_params, spg_opts);
svrg_obj_his, svrg_err_his = solveDMD_withSVRG(svrg_params, svrg_opts);


# Plot results
# ------------------------------------------------------------------------------
dirname = @__DIR__

pg_inds = [0; n * ((1:length(pg_obj_his)) + cumsum(pg_lns_his))];

spg_inds = [0; tau * (1:length(spg_obj_his)) .+ n];
svrg_inds = [0; tau * (1:length(svrg_obj_his)) .+ n];

pg_obj_his = [init_obj; pg_obj_his];
spg_obj_his = [init_obj; spg_obj_his];
svrg_obj_his = [init_obj; svrg_obj_his];
min_obj = (1 - 1e-8) * min([pg_obj_his; spg_obj_his; svrg_obj_his]...);

maxind = maximum(spg_inds)

ii = pg_inds .<= maxind

pg_inds = pg_inds[ii]
pg_obj_his = pg_obj_his[ii]

ph = plot(pg_inds, pg_obj_his .- min_obj, label="PG");
plot!(spg_inds, spg_obj_his .- min_obj, label="SPG")
plot!(svrg_inds, svrg_obj_his .- min_obj, label="SVRG")
xaxis!("number of solved subproblems")
yaxis!("objective function",:log10)
title!("Algorithm Comparison")

#  save figure
mkpath(dirname * "/figures")
fname = dirname * "/figures/alg_comparison.pdf";
savefig(ph,fname)
