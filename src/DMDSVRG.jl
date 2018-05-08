#==========================================================
    SVRG Solver for DMD
==========================================================#

using StatsBase

type SVRG_options
    itm::Int64
    tol::Float64
    batch_size::Int64
    step_size::Float64
    update_nu_every::Int64
    update_obj_every::Int64
    ifstats::Bool
    show_his::Bool
    print_frequency::Int64
    prox::Function
end

function SVRG_options(itm::Int64,tol::Float64,batch_size::Int64,
                      step_size::Float64,update_nu_every::Int64,
                      update_obj_every::Int64,ifstats::Bool,
                      show_his::Bool,
                      print_frequency::Int64;
                      prox::Function = SVRGnullprox!)
    return SVRG_options(itm,tol,batch_size,step_size,update_nu_every,
                        update_obj_every,ifstats,show_his,print_frequency,prox)
end

function SVRGnullprox!(x)
    return
end

function SVRG_solve_DMD!(vars, params, svars, options)
    # rename variables
    itm = options.itm;
    tol = options.tol;
    tau = options.batch_size;
    mu  = options.step_size;
    ifstats = options.ifstats;
    stats = OptimizerStats(itm,ifstats)
    pf  = options.print_frequency;
    uf  = options.update_nu_every;
    of = options.update_obj_every;
    prox = options.prox
    n   = params.n;
    k   = params.k;
    VPSolver! = svars.VPSolver!
    VPSolver_subset! = svars.VPSolver_subset!
    # pre-allocated variables
    ind = collect(1:n);
    indtau = collect(1:tau);
    sgr = zeros(2*k);
    dgr = zeros(2*k);
    prox(vars.alphar)
    alphar_old = copy(vars.alphar);
    # store all the gradient
    Gc  = zeros(vars.B);
    pg  = convert(Ptr{Float64}, pointer(Gc));
    gr  = Array{Array{Float64,1}}(n);
    sr  = sizeof(Float64);
    for id = 1:n
        gr[id] = unsafe_wrap(Array, pg, 2*k);
        pg += sr*2*k;
    end
    # calculate the full gradient of alpha once
    VPSolver!(vars,params,svars)
    for id = 1:n
        alphagrad_sub!(gr[id], vars, params, id);
        BLAS.axpy!(1.0,gr[id],sgr);
    end
    obj = DMDObj(vars, params; updatephi=true, updateR=true);
    err = 1.0;
    noi = 0;
    nu  = mu;
    updateOptimizerStats!(stats,obj,err,noi,ifstats)
    @show obj;
    for noi = 1:itm
        sample!(ind,indtau,replace=false);
        fill!(dgr, 0.0);
        
        # project out all for this subset
        VPSolver_subset!(vars,params,svars,indtau)

        # remove old gradient for each index and add new
        for j = 1:tau
            id = indtau[j];
            BLAS.axpy!(-1.0,gr[id],dgr);
            alphagrad_sub!(gr[id], vars, params, id);
            BLAS.axpy!( 1.0,gr[id],dgr);
        end

        # update α
        BLAS.axpy!(-nu/tau, dgr, vars.alphar);
        BLAS.axpy!(-nu/n, sgr, vars.alphar);

        ##### add projection of the real part #####
        prox(vars.alphar)

        ##### disable BB line search #####
        # if noi > itm
        #     BLAS.axpy!(-1.0,vars.alphar,alphar_old);
        #     nu = -(dot(alphar_old,alphar_old)/dot(alphar_old,dgr))*tau/n;
        #     (isnan(nu)|| nu < 0.0) && (nu = mu/(noi + 1));
        #     err = vecnorm(alphar_old);
        # end

        ##### using diminishing step size #####
        nu = mu/sqrt(div(noi,uf) + 1);
        err = vecnorm(vars.alphar - alphar_old);

        # @show nu
        copy!(alphar_old, vars.alphar);

        # update gradient
        BLAS.axpy!(1.0, dgr, sgr);

        # convergent history (this is a rough estimate)
        if (noi % of ==0)
            obj = DMDObj(vars, params; updatephi=true, updateR=true);
        else
            updatephimat!(vars.phi, params.t, vars.alpha);
        end
        updateOptimizerStats!(stats,obj,err,noi,ifstats)
        # print information
        (options.show_his && noi % pf == 0) &&
            @printf("iter %7d, obj %1.5e, err %1.5e\n", noi, obj, err);
        err < tol && break;

    end

    # update all of the b columns to get fit for final alpha
    VPSolver!(vars,params,svars)
    obj = DMDObj(vars, params; updatephi=true, updateR=true);
    updateOptimizerStats!(stats,obj,err,noi,ifstats)

    # return stats
    return stats
    
end
