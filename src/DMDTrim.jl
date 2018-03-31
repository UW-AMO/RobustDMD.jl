#==========================================================
    Trim Solver for DMD
==========================================================#
type Trim_options
    itm::Int64
    tol::Float64
    step_size::Float64
    ifstats::Bool
    show_his::Bool
    print_frequency::Int64
    prox::Function
end

function Trim_options(itm::Int64,tol::Float64,step_size::Float64,
                      ifstats::Bool,show_his::Bool,
                      print_frequency::Int64; 
                      prox::Function = Trimnullprox!)
    return Trim_options(itm,tol,step_size,ifstats,show_his,
                        print_frequency,prox)
end

function Trimnullprox!(x)
    return
end

function Trim_solve_DMD!(vars, params, svars, options)
    # rename variables
    itm = options.itm;
    tol = options.tol;
    mu  = options.step_size;
    pf  = options.print_frequency;
    ifstats = options.ifstats;
    prox = options.prox
    stats = OptimizerStats(itm,ifstats);
    n   = params.n;
    k   = params.k;
    h   = params.h;
    VPSolver! = svars.VPSolver!
    # pre-allocated variables
    ind = collect(1:n);
    sgr = zeros(2*k);
    dgr = zeros(2*k);
    prox(vars.alphar)
    alphar_old = copy(vars.alphar);
    # store all the gradient
    Gc  = zeros(vars.B);
    fb  = zeros(n);
    idh = zeros(Int64,n);
    pg  = convert(Ptr{Float64}, pointer(Gc));
    gr  = Array(Array{Float64,1}, n);
    sr  = sizeof(Float64);
    for id = 1:n
        gr[id] = unsafe_wrap(Array, pg, 2*k);
        pg += sr*2*k;
    end

    # compute initial loss for each column
    updatephimat!(vars.phi, params.t, vars.alpha)
    VPSolver!(vars,params,svars)
    updateResidual!(vars, params)
    for id = 1:n
        fb[id] = DMDObj_sub(vars, params, id; updatephi = false, updateR = false);
    end

    # sort columns by badness of fit
    sortperm!(idh, fb);

    # calculate the gradient of alpha once    
    for ind = 1:h
        id = idh[ind];
        alphagrad_sub!(gr[id], vars, params, id);
        BLAS.axpy!(1.0,gr[id],sgr);
    end

    # starting error for best h columns
    err = 1.0
    obj = 0.0;
    for ind = 1:h
        id = idh[ind];
        obj += fb[id];
    end
    noi = 0
    updateOptimizerStats!(stats,obj,err,noi,ifstats)    
    
    @show obj;
    #@show sgr;
    for noi = 1:itm
        nu = mu;
        # update alpha (take a step)
        copy!(alphar_old, vars.alphar);
        BLAS.axpy!(-nu, sgr, vars.alphar);
        # project
        prox(vars.alphar)
        # err
        err = vecnorm(alphar_old-vars.alphar);
        # update gradient
        copy!(dgr, sgr);
        fill!(sgr, 0.0);
        # project out b
        updatephimat!(vars.phi, params.t, vars.alpha)
        VPSolver!(vars,params,svars)
        updateResidual!(vars, params)
        # get norms by column
        for id = 1:n
            fb[id] = bfunc_sub(vars.br[id], vars, params, id)
        end
        sortperm!(idh, fb);
        # compute gradient based on h best
        for ind = 1:h
            id = idh[ind];
            alphagrad_sub!(gr[id], vars, params, id);
            BLAS.axpy!(1.0,gr[id],sgr);
        end
        BLAS.axpy!(-1.0, sgr, dgr);
        # @show norm(dgr);
        # convergent history
        obj = 0.0;
        for ind = 1:h
            id = idh[ind];
            obj += fb[id]
        end
        updateOptimizerStats!(stats,obj,err,noi,ifstats)        
        # print information
        (options.show_his && noi % pf == 0) &&
            @printf("iter %5d, obj %1.5e, err %1.5e\n", noi, obj, err);

        err < tol && break
        
    end

    return stats
    
end
