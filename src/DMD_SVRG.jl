# SVRG solver for large scale DMD problem
mutable struct DMD_SVRG_Options{T<:AbstractFloat}
	τ::Integer	# sample size
	η::T		# step size
	itm::Integer
	tol::T
	ptf::Integer
	# pre-allocate variables
	fars::Vector{T}
	Gars::Matrix{T}
	gars::Array{Vector{T},1}
end

# constructor of the SVRG options
function DMD_SVRG_Options(τ, η, params; itm=1000, tol=1e-5, ptf=100)
	k = params.k;
	T = typeof(real(params.X[1]));

	tol  = T(tol);

	fars = zeros(T, n);
	Gars = zeros(T, 2*k, n);
	gars = col_view(Gars);

	return DMD_SVRG_Options(τ, η, itm, tol, ptf, fars, Gars, gars)
end

# SVRG solver
function solveDMD_withSVRG(vars, params, svars, opts)
	# load all the variables
	τ    = opts.τ;
	η    = opts.η;
	n    = params.n;
	k    = params.k;
	T    = typeof(real(params.X[1]));

	ar   = vars.ar;
	dar  = zeros(T, 2*k);
	ind  = collect(1:n);

	fars = opts.fars;
	gars = opts.gars;
	
	tfar = T(0.0); tgar = zeros(T, 2*k);
	dfar = T(0.0); dgar = zeros(T, 2*k);
	rfar = T(0.0); rgar = zeros(T, 2*k);

	itm  = opts.itm;
	tol  = opts.tol;
	ptf  = opts.ptf;

	obj_his = zeros(T, itm);
	err_his = zeros(T, itm);

	# first full step
	# initialize the gradient (might take a while)
	update_P!(vars, params);
	update_PQR!(vars, params, svars);
	update_B!(vars, params, svars);
	update_R!(vars, params);
	for i = 1:n
		fars[i] = objective(ar, vars, params, svars, i,
			updateP=false, updateB=false, updateR=false);
		gradient!(ar, gars[i], vars, params, svars, i,
			updateP=false, updateB=false, updateR=false);
	end
	tfar = sum(fars);
	sum!(tgar, opts.Gars); @show vecnorm(gars[1]);
	# update alpha
	copy!(dar, tgar); scale!(dar, η/n);
	broadcast!(-, ar, ar, dar);
	
	err  = vecnorm(dar);
	noi  = 0;

	while err ≥ tol
		# update information corresponding to alpha
		update_P!(vars, params);
		update_PQR!(vars, params, svars);
		# random sample columns
		shuffle!(ind); fill!(dgar, T(0.0)); dfar = T(0.0);
		for i = 1:τ
			id = ind[i];
			# calcualte the objecitve
			rfar = objective(ar, vars, params, svars, id,
				updateP=false, updateB=true, updateR=true);
			dfar = dfar + rfar - fars[id];
			fars[id] = rfar;
			# calcualte the gradient
			gradient!(ar, rgar, vars, params, svars, id,
				updateP=false, updateB=false, updateR=false);
			broadcast!(+, dgar, dgar, rgar);
			broadcast!(-, dgar, dgar, gars[id]);
			copy!(gars[id], rgar);
		end
		# update alpha
		copy!(dar, tgar); scale!(dar, η/n);
		BLAS.axpy!(η/τ, dgar, dar);
		broadcast!(-, ar, ar, dar);
		# update tfar and tgar
		tfar = tfar + dfar;
		broadcast!(+, tgar, tgar, dgar);

		# update information
		err = vecnorm(dar);
		noi = noi + 1;
		obj_his[noi] = tfar;
		err_his[noi] = err;

		# print information
		noi % ptf == 0 && @printf("iter %5d, obj %1.2e, err %1.2e\n",
			noi, tfar, err);
		noi ≥ itm && break;
	end
	return obj_his[1:noi], err_his[1:noi]
end