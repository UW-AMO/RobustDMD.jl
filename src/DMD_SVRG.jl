# SVRG solver for large scale DMD problem
mutable struct DMD_SVRG_Options{T<:AbstractFloat}
	τ::Integer	# sample size
	η::T		# step size
	itm::Integer
	tol::T
	ptf::Integer
	# pre-allocate variables
	ind ::Vector{Integer} # random indices
	fs  ::Vector{T}
	Gs  ::Matrix{T}
	gs  ::Array{Vector{T},1}
	dar ::Vector{T} # difference between current and previous iterators
	tgar::Vector{T}	# temporary variable to store the partial gradient of alpha
	pgar::Vector{T}	# approximate gradient of alpha
	dgar::Vector{T}	# different gradient of alpha
end

# constructor of the SVRG options
function DMD_SVRG_Options(τ, η, params; itm=1000, tol=1e-5, ptf=100)
	k = params.k;
	T = typeof(real(params.X[1]));
	tol  = T(tol);
	ind  = collect(1:params.n);
	fs   = zeros(T, n);
	Gs   = zeros(T, k, n);
	gs   = col_view(Gs);
	dar  = zeros(T, k);
	tgar = zeros(T, k);
	pgar = zeros(T, k);
	dgar = zeros(T, k);
	return DMD_SVRG_Options(τ, η, itm, tol, ptf, ind, fs, Gs, gs, dar, tgar, pgar, dgar)
end

# SVRG solver
function solveDMD_withSVRG(vars, params, svars, opts)
	# load all the variables
	τ    = opts.τ;
	η    = opts.η;
	n    = params.n;

	ar   = vars.ar;
	dar  = opts.dar;
	ind  = opts.ind;
	fs   = opts.fs;
	gs   = opts.gs;
	tf   = T(0.0);
	df   = T(0.0);

	tgar = opts.tgar;
	pgar = opts.pgar;
	dgar = opts.dgar;

	itm  = opts.itm;
	tol  = opts.tol;
	ptf  = opts.ptf;

	T = typeof(real(params.X[1]));

	# initialize the gradient (might take a while)
	update_P!(vars, params);
	update_PQR!(vars, params, svars);
	update_B!(vars, params, svars);
	for i = 1:n
		fs[i] = objective(ar, vars, params, svars, i,
			updateP=false, updateB=false, updateR=false);
		gradient!(ar, gs[i], vars, params, svars, i,
			updateP=false, updateB=false, updateR=false);
	end
	obj  = sum(fs);
	pgar = sum(opts.Gs, 2);
	copy!(dar, pgar); scale!(dar, η/n);
	broadcast!(-, ar, ar, dar);
	
	err  = vecnorm(dar);
	noi  = 0;

	while err ≥ tol
		# update information corresponding to alpha
		update_P!(vars, params);
		update_PQR!(vars, params, svars);
		# random sample columns
		randperm!(ind); fill!(dgar, T(0.0)); df = T(0.0);
		for i = 1:τ
			id = ind[i];
			tf = objective(ar, vars, params, svars, id,
				updateP=false, updateB=true, updateR=true);
			df = df + tf;
			df = df - fs[id];
			fs[id] = tf;
			gradient!(ar, tgar, vars, params, svars, id,
				updateP=false, updateB=false, updateR=false);
			broadcast!(+, dgar, dgar, tgar);
			broadcast!(-, dgar, dgar, gs[id]);
			copy!(gs[id], tgar);
		end
		# update alpha
		copy!(dar, pgar); scale!(dar, η/n);
		BLAS.axpy!(η/τ, dgar, dar);
		broadcast!(-, ar, ar, dar);

		# update information
		err = vecnorm(dar);
		obj = obj + df;
		noi = noi + 1;

		# print information
		noi % ptf == 0 && @printf("iter %5d, obj %1.2e, err %1.2e\n",
			noi, obj, err);
		noi ≥ itm && break;
	end
end