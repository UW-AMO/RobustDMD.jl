export DMD_SVRG_Options, solveDMD_withSVRG

using StatsBase, Printf

# SVRG solver for large scale DMD problem
mutable struct DMD_SVRG_Options
    tau::Integer	# sample size
    eta::AbstractFloat # step size
    itm::Integer 
    tol::AbstractFloat
    ptf::Integer
    true_obj::Bool
    prox::Function
end

# constructor of the SVRG options
function DMD_SVRG_Options(tau, eta; itm=1000, tol=1e-5, ptf=100,
                          true_obj=false,
                          prox=prox_null::Function)

    return DMD_SVRG_Options(tau, eta, itm, tol, ptf, true_obj, prox)
end

# SVRG solver
function solveDMD_withSVRG(params, opts)
    # load all the variables
    tau    = opts.tau;
    eta    = opts.eta;
    prox = opts.prox;
    n    = params.n;
    k    = params.k;
    T    = typeof(real(params.X[1]));

    ar   = params.ar;
    arold = copy(ar);
    prox(ar);
    dar  = zeros(T, 2 * k);
    ind  = collect(1:n);

    fars = zeros(T, n)
    Gars = zeros(T, 2 * k, n)
    gars = Array{typeof(view(Gars,:,1))}(undef,n)
    for i = 1:n
        gars[i] = view(Gars,:,i)
    end
    
    tfar = T(0.0); tgar = zeros(T, 2 * k);
    dfar = T(0.0); dgar = zeros(T, 2 * k);
    rfar = T(0.0); rgar = zeros(T, 2 * k);

    itm  = opts.itm;
    tol  = opts.tol;
    ptf  = opts.ptf;
    
    obj_his = zeros(T, itm);
    err_his = zeros(T, itm);

    # first full step
    # initialize the gradient (might take a while)
    for id = 1:n
        fars[id] = abFunc(params, id);
        abGrad(gars[id], params, id);
    end

    tfar = sum(fars);
    sum!(tgar, Gars);
    # update alpha
    copyto!(dar, tgar); dar .*= eta / n;
    copyto!(arold, ar);
    ar .-= dar;

    prox(ar);
    BLAS.axpy!(-1.0, ar, arold)
    err  = sqrt(sum(abs2, arold));
    noi  = 0;

    ind2 = collect(1:tau);

    while err >= tol

        # random sample columns
        sample!(ind, ind2,replace=false);
        fill!(dgar, T(0.0)); dfar = T(0.0);
        for i = 1:tau
            id = ind2[i];
            # calculate the objecitve
            rfar = abFunc(params, id);
            dfar = dfar + rfar - fars[id];
            fars[id] = rfar;
            # calculate the gradient
            dgar .-= gars[id];
            abGrad(gars[id], params, id)
            dgar .+= gars[id];
        end
        # update alpha
        copyto!(dar, tgar); dar .*= eta / n;
        BLAS.axpy!(eta / tau, dgar, dar);
        copyto!(arold, ar);
        ar .-= dar;
        prox(ar);

        # update tfar and tgar
        if opts.true_obj
            for id = 1:n
                fars[id] = abFunc(params, id);
            end
            tfar = sum(fars);
        else
            tfar  += dfar;
        end
        tgar .+= dgar;

        # update information
        BLAS.axpy!(-1.0, ar, arold)
        err = sqrt(sum(abs2, arold));
        noi = noi + 1;
        obj_his[noi] = tfar;
        err_his[noi] = err;

        # print information
        noi % ptf == 0 && @printf("iter %5d, obj %1.2e, err %1.2e\n",
                                  noi, tfar, err);
        noi >= itm && break;
    end
    return obj_his[1:noi], err_his[1:noi]
end
