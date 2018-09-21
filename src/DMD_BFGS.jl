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
	Har  = diagm(fill(σ, 2*k));

	return DMD_BFGS_Options(itm, tol, ptf, ar, nar,
		gar, ngar, par, sar, yar, Har)
end

# BFGS solver
function solveDMD_withBFGS(vars, params, svars, opts)
	# load all the variables
	T    = typeof(real(params.X[1]));
	itm  = opts.itm;
	tol  = opts.tol;
	ptf  = opts.ptf;

	ar   = opts.ar; copy!(ar, vars.ar);
	nar  = opts.nar;
	gar  = opts.gar;
	ngar = opts.ngar;
	par  = opts.par;
	sar  = opts.sar;
	yar  = opts.yar;
	Har  = opts.Har;

	# first gradient step
	obj = objective(ar, vars, params, svars);
	gradient!(ar, gar, vars, params, svars,
		updateP=false, updateB=false, updateR=false);
	err = vecnorm(gar, Inf);
	noi = 0;

	@show obj
	@show err

	obj_his = zeros(T, itm);
	err_his = zeros(T, itm);

	while err ≥ tol
		# calculate the direction
		BLAS.gemv!('N', T(-1.0), Har, gar, T(0.0), par);

		# decend line search
		η = 1.0;
		copy!(nar, ar); BLAS.axpy!(η, par, nar);
		nobj = objective(nar, vars, params, svars);
		while nobj ≥ obj
			η *= 0.5;
			η < 1e-10 && break;
			copy!(nar, ar); BLAS.axpy!(η, par, nar);
			nobj = objective(nar, vars, params, svars);
		end

		# update gradient
		gradient!(nar, ngar, vars, params, svars,
			updateP=false, updateB=false, updateR=false);

		# update differences
		broadcast!(-, sar, nar, ar); copy!(ar, nar);
		broadcast!(-, yar, ngar, gar); copy!(gar, ngar);

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
		obj = nobj;
		err = vecnorm(par, Inf);
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