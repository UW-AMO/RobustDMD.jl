# BFGS solver for DMD problem
mutable struct DMD_BFGS_Options{T<:AbstractFloat}
	itm::Integer
	tol::T
	ptf::Integer
	# pre-allocate variables
	ar  ::Vector{T}
	nar ::Vector{T}
	gar ::Vector{T}
	ngar::Vector{T}
	par ::Vector{T}
	sar ::Vector{T}
	yar ::Vector{T}
	Har ::Matrix{T}
end

# constructor for BFGS options
function DMD_BFGS_Options(σ, params; itm=100, tol=1e-5, ptf=10)
	T = typeof(real(params.X[1]));
	tol = T(tol);
	k = params.k;

	ar   = zeros(T, 2*k);
	nar  = zeros(T, 2*k);
	gar  = zeros(T, 2*k);
	ngar = zeros(T, 2*k);
	par  = zeros(T, 2*k);
	sar  = zeros(T, 2*k);
	yar  = zeros(T, 2*k);
	Har  = diagm(0 => fill(σ, 2*k));

	return DMD_BFGS_Options(itm, tol, ptf, ar, nar,
		gar, ngar, par, sar, yar, Har)
end

# BFGS solver
function solveDMD_withBFGS(params, opts)
	# load all the variables
	T    = typeof(real(params.X[1]));
	itm  = opts.itm;
	tol  = opts.tol;
	ptf  = opts.ptf;

	ar   = params.ar;
	nar  = opts.nar;
	gar  = opts.gar;
	ngar = opts.ngar;
	par  = opts.par;
	sar  = opts.sar;
	yar  = opts.yar;
	Har  = opts.Har;

	# first gradient step
	nobj = aBFunc(params);
	aBGrad(ngar, params);
	err = norm(ngar, Inf);
	noi = 0;

	obj_his = zeros(T, itm);
	err_his = zeros(T, itm);

	while err ≥ tol
		# calculate the direction
		BLAS.gemv!('N', T(-1.0), Har, ngar, T(0.0), par);
		# decend line search
		η = 1.0;
		copyto!(nar, ar); BLAS.axpy!(η, par, ar);
		obj = aBFunc(params);
		while obj ≥ nobj
			η *= 0.5;
			η < 1e-10 && break;
			copyto!(ar, nar); BLAS.axpy!(η, par, ar);
			obj = aBFunc(params);
		end

		# update gradient
		aBGrad(gar, params);
		# update differences
		sar .= ar - nar; copyto!(nar, ar);
		yar .= gar - ngar; copyto!(ngar, gar);

		# update Hessian approximation
		ρ = T(1.0)/dot(yar, sar);
		μ = sum(abs2, sar);
		ρ = min(T(10.0), ρ*μ)/μ;
		if ρ > 0.0
			BLAS.gemv!('N', T(1.0), Har, yar, T(0.0), par);
			β = dot(yar, par)*ρ^2 + ρ;
			BLAS.ger!(-ρ, sar, par, Har);
			BLAS.ger!(-ρ, par, sar, Har);
			BLAS.ger!( β, sar, sar, Har);
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
		noi ≥ itm && break;
	end

	return obj_his[1:noi], err_his[1:noi]
end