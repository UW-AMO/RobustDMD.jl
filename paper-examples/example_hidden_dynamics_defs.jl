# This file defines the hidden dynamics system problem
# with (1) background noise and sparse large
# deviations, (2) "broken sensor" noise, and
# (3) an added "bump". L2, robust (huber), and
# robust (trimming) fits are
# run for several background noise levels
# and the quality of the resulting eigenvalues is
# returned

using Optim

################################################################
################################################################
################################################################
# Define some helper routines


function hidden_dynamics_system_data(;nt=2^9,
                                     nx=300,k1=1.0,
                                     omega1=1.0,
                                     gamma1=1.0,
                                     k2=0.4,
                                     omega2=3.7,
                                     gamma2=-0.2)
################################################
# Generate data for hidden dynamics system
#
# z(t,x) = exp(gamma1*t)*sin(k1*x-omega1*t) + 
#      exp(gamma2*t)*sin(k2*x-omega2*t) + 
#
# as gamma1 is positive and gamma2 is negative,
# the mode for gamma2 becomes hidden, especially
# with significant noise.
# 
################################################

    evals = [gamma1 + im*omega1; gamma1 - im*omega1;
             gamma2 + im*omega2; gamma2 - im*omega2]
    
    x = collect(range(0,stop=15.0,length=nx))
    t = complex(collect(range(0,stop=2*pi,length=nt)))
    
    z = complex(exp.(gamma1*t) .* sin.(k1*transpose(x) .- omega1*t) + 
         exp.(gamma2*t) .* sin.(k2*transpose(x) .- omega2*t))
    
    return z, evals, t, x
end



function make_corrupt_data_sparse_spikes(xclean,sigma,mu,p)
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

function make_corrupt_data_bump(xclean,x,t,sigma,A,w)
    #------------------------------------------
    # Generate Corrupted Data - bump
    #------------------------------------------
    #
    # sigma - background noise
    # A - bump amplitude
    # w - bump width
    
    m, n = size(xclean)

    i1 = rand(1:m)
    j1 = rand(1:n)
    x1 = x[j1]
    t1 = t[i1]
    
    dx = x[2]-x[1]
    dt = t[2]-t[1]

    noise = (sigma*randn(m,n) .+
             A*exp.( real(-(t .- t1).^2/(dt*w)^2 .- (transpose(x) .- x1).^2/(dx*w)^2 )))
    xdat = xclean .+ noise
    

    return xdat
end


function make_corrupt_data_broken_sensors(xclean,sigma,mu,p)
    #---------------------------------------------------------
    # Generate Corrupted Data - "broken sensors"
    #---------------------------------------------------------
    #
    # sigma - background noise
    # mu - spike size
    # p - spike frequency

    m, n = size(xclean)
    
    ncol = Integer(round(n*p))
    icols = sample(1:n,ncol,replace=false)

    noise = sigma*randn(m,n)
    noise[:,icols] .+= mu*randn(m,length(icols))
    xdat = xclean .+ noise

    return xdat
end


function dofits(xdat,t,eigs,kappa,optionsl2,optionshuber,optionstrim,
                nkeep)
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
    res = solveDMD_withOptimBFGS(l2params)

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
    res = solveDMD_withOptimBFGS(huberparams)
    #----------------------------------------------------  
    # l2 trimming trial
    #----------------------------------------------------
    #println("running l2 trimming experiment...");
    # loss functions
    lossf = (z) -> l2_func(z);
    lossg = (z) -> l2_grad!(z);

    trimparams = DMDParams(k, xdat, t, lossf, lossg,
                           inner_directl2=true,nkeep=nkeep);

    copyto!(trimparams.a,ainit)
    
    # apply solver
    obj_his3, err_his3 = solveDMD_withPG(trimparams, optionstrim);
    res = solveDMD_withOptimBFGS(trimparams)
    #--------------------------------------------------------------------
    # Evaluate Errors and Get Best Eigenvalue Permutation
    #--------------------------------------------------------------------

    if any(isnan,l2params.a)
        err1 = Inf
        p1 = 1:k
    else
        err1, p1 = besterrperm_wi(l2params.a, eigs);
    end
    if any(isnan,huberparams.a)
        err2 = Inf
        p2 = 1:k
    else
        err2, p2 = besterrperm_wi(huberparams.a, eigs);
    end
    if any(isnan,trimparams.a)
        err3 = Inf
        p3 = 1:k
    else
        err3, p3 = besterrperm_wi(trimparams.a, eigs);
    end
    err4, p4 = besterrperm_wi(a0, eigs);

    return (err1, err2, err3, err4, l2params.a[p1], huberparams.a[p2], 
            trimparams.a[p3], a0[p4])
    
end

