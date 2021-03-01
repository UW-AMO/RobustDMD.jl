export DMD_PG_Options, solveDMD_withPG

# This file is available under the terms of the MIT License

mutable struct DMD_PG_Options{T<:AbstractFloat}
    itm::Integer
    tol::T
    ptf::Integer

    prox::Function
end

function DMD_PG_Options(;itm=100,tol=1e-5,ptf=10,
                        prox=prox_null::Function)

    return DMD_PG_Options(itm,tol,ptf,prox)
end

function solveDMD_withPG(params, opts)
    # iteration parameters
    itm = opts.itm;
    tol = opts.tol;
    ptf = opts.ptf;
    # prox function
    prox = opts.prox;
    # problem dimensions
    m = params.m;
    n = params.n;
    k = params.k;
    
    T = typeof(real(params.X[1]))

    # pre-allocate the variables
    gar  = zeros(T,k<<1);
    garp1 = zeros(T,k<<1);
    ar  = copy(params.ar);
    prox(ar)
    arm1 = copy(ar);

    function a_fg!(gar,arnew)
        copyto!(params.ar,arnew);
        return aBFuncGrad(gar,params)
    end

    function a_f(arnew)
        copyto!(params.ar,arnew);
        return aBFunc(params);
    end
    
    # initialize variables
    obj = a_fg!(gar, ar);
    err = T(1.0);
    noi = 0;

    if ( isnan(obj) || isinf(obj) )
        println("BAD INITIAL GUESS ",ar)
        println("ABORT")
        return
    end

    obj_his = zeros(itm)
    err_his = zeros(itm)

    alphastart = 10.0

    while err >= tol
        eta = descent_PG!(arm1, ar, gar, obj, a_f, prox,
                          alphastart=alphastart);
        prox(ar);

        obj = a_fg!(gar, ar);
        err = sqrt(sum(abs2,ar - arm1))/eta;
        #err = sqrt(sum(abs2,ar - arm1));        
        #err = vecnorm(ar - arm1)
        copyto!(arm1, ar);
        noi += 1;
        # @show gar;
        # @show ar;
        (noi % ptf == 0) && @printf("PG: iter %4d, obj %1.6e, err %1.6e\n",
                                                    noi, obj, err);

        obj_his[noi] = obj
        err_his[noi] = err
        noi >= itm && break;
        alphastart = min(2*eta,10.0)
    end
    
    return obj_his[1:noi], err_his[1:noi]
    
end

#--------------------------
# Descent Line Search
#--------------------------
function descent_PG!(xm1, x, g, obj, f, prox;
                     alphamin = 1e1*eps(real(eltype((x)))),
                     tol = 1e2*eps(real(eltype(x))),
                     alphastart = 10.0)
    T = real(eltype(x))
    n = length(x);
    alpha = T(alphastart);
    noi = 0;
    for i = 1:n
        x[i] = xm1[i] - alpha*g[i];
    end
    prox(x);
    objp1 = f(x);
    while objp1 > obj || isnan(objp1)
        alpha *= T(0.5);
        for i = 1:n
            x[i] = xm1[i] - alpha*g[i];
        end
        prox(x);
        objp1 = f(x);
        noi += 1;
        # noi ≥ 5 && break;
        alpha < alphamin && break;
    end
    # @show noi, a
    if (objp1 > obj || isnan(objp1))
        alpha = T(0)
        for i = 1:n
            x[i] = xm1[i]
        end
    end
    return alpha;
end
