export DMD_BFGS_Options, solveDMD_withBFGS

# BFGS solver for DMD problem
mutable struct DMD_BFGS_Options{T<:AbstractFloat}
    itm::Integer
    tol::T
    ptf::Integer
    sigma::T
end

# constructor for BFGS options
function DMD_BFGS_Options(;sigma=10.0, itm=100, tol=1e-5,
                          ptf=10)
    
    return DMD_BFGS_Options(itm, tol, ptf, sigma)
end

# BFGS solver
function solveDMD_withBFGS(params,opts)
    
    # load all the variables
    T    = typeof(real(params.X[1]));
    itm  = opts.itm;
    tol  = opts.tol;
    ptf  = opts.ptf;

    ar   = params.ar;
    sigma = T(opts.sigma)

    k = params.k
    
    nar  = zeros(T, 2*k);
    gar  = zeros(T, 2*k);
    ngar = zeros(T, 2*k);
    par  = zeros(T, 2*k);
    sar  = zeros(T, 2*k);
    yar  = zeros(T, 2*k);
    Har  = diagm(0 => fill(sigma, 2*k));
    
    # first gradient step
    nobj = aBFunc(params);
    aBGrad(ngar, params);
    err = norm(ngar, Inf);

    noi = 0;

    obj_his = zeros(T, itm);
    err_his = zeros(T, itm);

    while err >= tol
	# calculate the direction
	BLAS.gemv!('N', T(-1.0), Har, ngar, T(0.0), par);
	# decend line search
	eta = 1.0;
	copyto!(nar, ar); BLAS.axpy!(eta, par, ar);
	obj = aBFunc(params);
	while obj >= nobj
	    eta *= 0.5;
	    eta < 1e-12 && break;
	    copyto!(ar, nar); BLAS.axpy!(eta, par, ar);
	    obj = aBFunc(params);
	end

	# update gradient
	aBGrad(gar, params);
	# update differences
	sar .= ar - nar; copyto!(nar, ar);
	yar .= gar - ngar; copyto!(ngar, gar);

	# update Hessian approximation
	rho = T(1.0)/dot(yar, sar);
	mu = sum(abs2, sar);
	rho = min(T(10.0), rho*mu)/mu;
	if rho > 0.0
	    BLAS.gemv!('N', T(1.0), Har, yar, T(0.0), par);
	    beta = dot(yar, par)*rho^2 + rho;
	    BLAS.ger!(-rho, sar, par, Har);
	    BLAS.ger!(-rho, par, sar, Har);
	    BLAS.ger!( beta, sar, sar, Har);
	end

	# update information
	nobj = obj;
	err = norm(par, Inf);
	noi = noi + 1;

	obj_his[noi] = obj;
	err_his[noi] = err;

	# print information
	noi % ptf == 0 && @printf("iter %4d, obj %1.2e, err %1.2e\n",
			          noi, obj, err);
	noi >= itm && break;
        isnan(obj) && break;
    end

    return obj_his[1:noi], err_his[1:noi]
end
