# unit test the set up

include("../src/DMD_Util.jl");
include("../src/DMD_Type.jl");

using LowRankApprox

# Generate DMD Synthetic Data
#------------------------------------------------------------------------------
# dimensions
m = 100;    # temporal dimension
n = 100;    # spatial dimension
k = 3;      # number of modes
T = Float64;
# generate data
@time X, t, at, Bt = simDMD(m, n, k, T; seed=123456);

# obtain exact dmd and trap dmd estimates


dmdexactestimate(m,n,k,X,t)

@time a_exact, B_exact = dmdexactestimate(m,n,k,X,t,dmdtype="exact")
@time a_trap, B_trap = dmdexactestimate(m,n,k,X,t,dmdtype="trap")

@time a_exact_approx, B_exact_approx = dmdexactestimate(m,n,k,X,
                                                        t,dmdtype="exact",
                                                        uselowrank=true)
@time a_trap_approx, B_trap_approx = dmdexactestimate(m,n,k,X,
                                                        t,dmdtype="trap",
                                                        uselowrank=true)

@printf "%7.4e abs err evals w/ exactDMD\n" besterrperm(a_exact,at)
@printf "%7.4e abs err evals w/ trapDMD\n" besterrperm(a_trap,at)
@printf "%7.4e abs err evals w/ exactDMD (approximate)\n" besterrperm(
    a_exact_approx,at)
@printf "%7.4e abs err evals w/ trapDMD (approximate)\n" besterrperm(
    a_trap_approx,at)
