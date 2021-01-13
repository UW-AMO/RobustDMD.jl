# simple periodic system runner.
#
# This file defines the periodic system problem
# with background noise and sparse large
# deviations. Both (global) L2 and Robust fitting are
# run for several background noise levels
# and the quality of the resulting eigenvalues is
# recorded and written to file
#


using RobustDMD, Printf, Random, JLD

################################################################
################################################################
################################################################
# Define some helper routines


function periodic_system_data(;nt=128)
################################################
# Generate data for simple periodic linear system
#
# z(t) = exp(At)z0
# 
# where A has eigenvalues equal to +/- i
# 
################################################
    
    z0 = [1.0; 0.1]
    A = [1 -2; 1 -1]
    dt = 0.1

    evals = [im; -im]

    z = Array{Complex{Float64}}(undef,nt,2)
    t = Array{Complex{Float64}}(undef,nt)

    for m = 1:nt
        t[m] = dt*m
        ztemp = exp(A*t[m])*z0
        z[m,:] = transpose(ztemp)
    end

    return z, evals, t
end



function make_corrupt_data(xclean,sigma,mu,p)
    #------------------------------------------
    # Generate Corrupted Data - Spikes
    #------------------------------------------
    #
    # sigma - background noise
    # mu - spike size
    # p - spike frequency
    
    m, n = size(xclean)


    noise = sigma*randn(m,n) .+ mu*randn(m,n).*( rand(m,n) .< p )
    xdat = xclean .+ noise

    return xdat
end


function dofits(xdat,t,eigs,kappa,optionsl2,optionshuber)
    # this wraps up the calls to the DMD solver
    # Here we use the robust but generally slower
    # proximal gradient routine for both L2 and Huber
    # fits. On return, we provide the error for these
    # fits and the exact DMD error.

    
    k = length(eigs)
    
    #-------------------------------------------------------------
    # Initial Guess
    #-------------------------------------------------------------
    m,n = size(xdat);
    
    a0, B0 = dmdexactestimate(m, n, k, xdat, t);

    ainit, Binit = dmdexactestimate(m,n,k,xdat,t,dmdtype="trap")
    
    #----------------------------------------------------
    # l2 Trial
    #----------------------------------------------------
    #println("running l2 experiment...");
    # loss functions
    lossf = (z) -> l2_func(z);
    lossg = (z) -> l2_grad!(z);

    l2params = DMDParams(k, xdat, t, lossf, lossg,
                         inner_directl2=true);
    copyto!(l2params.a,ainit)
    
    # apply solver
    obj_his1, err_his1 = solveDMD_withPG(l2params, optionsl2);

    #----------------------------------------------------
    # huber Trial
    #----------------------------------------------------
    #println("running huber experiment...");
    # loss functions
    lossf = (z) -> huber_func(z,kappa);
    lossg = (z) -> huber_grad!(z,kappa);

    huberparams = DMDParams(k, xdat, t, lossf, lossg);
    copyto!(huberparams.a,ainit)
    
    # apply solver
    obj_his2, err_his2 = solveDMD_withPG(huberparams, optionshuber);
    
    #--------------------------------------------------------------------
    # Evaluate Errors and Get Best Eigenvalue Permutation
    #--------------------------------------------------------------------
    
    err1, p1 = besterrperm_wi(l2params.a, eigs);
    err2, p2 = besterrperm_wi(huberparams.a, eigs);
    err3, p3 = besterrperm_wi(a0, eigs);

    return err1, err2, err3, l2params.a[p1], huberparams.a[p2], 
    a0[p3], l2params.B[p1,:], huberparams.B[p2,:], B0[p3,:], obj_his1,
    err_his1, obj_his2, err_his2
    
end

################################################################
################################################################
################################################################
# Actually run the tests...


# make data

xclean, eigs, t = periodic_system_data()

# solver parameters

function prox_mr(ar)
    # project for numerical stability
    k = length(ar) >> 1;
    for i = 1:k
        ar[i<<1-1] = min(1.0, ar[i<<1-1]);
    end
end

itm = 1000;
tol = 1e-7;
ptf = itm+1;
optsl2 = DMD_PG_Options(itm=itm, tol=tol, ptf=ptf, prox=prox_mr);
optshub = DMD_PG_Options(itm=itm, tol=tol, ptf=ptf, prox=prox_mr);


# test parameters

seed = 8675309
Random.seed!(seed)

p = 0.05 # add spikes at this percentage of data points
k = 2 # rank of fit

nj = 5 # number of different levels of background noise
ntest = 200

errs = Array{Float64}(undef,3,ntest,nj)
eigs_comp = Array{Complex{Float64}}(undef,k,3,ntest,nj)
params = Array{Float64}(undef,4,nj)

@printf "==================================================\n\n"
@printf "Running the simple periodic example... \n\n"

for j = 1:nj
    @printf "starting noise level %d of %d\n" j nj
    sigma = 10.0^(-6+j) # noise floor
    mu = 1.0 # size of spikes
    kappa = 5*sigma
    nu = sqrt(mu)
    params[1,j] = sigma
    params[2,j] = mu
    params[3,j] = p
    params[4,j] = kappa
    
    Threads.@threads for i = 1:ntest
        xdat = make_corrupt_data(xclean,sigma,mu,p)

        (err1, err2, err3, eig1, eig2, eig3, b1, b2, b3,
         obj_his1, err_his1, obj_his2, err_his2) = (
             dofits(xdat,t,eigs,kappa,optsl2,optshub))

        #@printf "test %d %d %d\n"  j l i
        #@printf "errs %e %e %e\n" err1 err2 err3
        errs[1,i,j] = err1
        errs[2,i,j] = err2
        errs[3,i,j] = err3

        eigs_comp[:,1,i,j] = eig1[:]
        eigs_comp[:,2,i,j] = eig2[:]
        eigs_comp[:,3,i,j] = eig3[:]

    end
end

fname = "results/simple_periodic_pg_out.jld"

@printf "writing results to file %s\n" fname

# save the results to file

mkpath("results")
file = jldopen(fname,"w")
file["errs"] = errs
file["eigs_comp"] = eigs_comp
file["params"] = params
close(file)

@printf "\n ...done\n"

@printf "==================================================\n"


