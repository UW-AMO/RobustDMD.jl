#====================================================================
    BFGS Solver
    -----------------------------------------------------------------
    first order semi-newton method, efficiently solving
    ill-conditioned objective functions
    -----------------------------------------------------------------
    input:
        func: objective function
        grad: gradient function
        x0: initial guess
        options: parameters control the iteration
    output:
====================================================================#

type BFGS_options
    itm::Int64
    tol::Float64
    ifstats::Bool
    warm_start::Bool    
    show_history::Bool
    ptf::Int64
end

type BFGS_vars
    x ::Array{Float64,1}
    x⁺::Array{Float64,1}
    g ::Array{Float64,1}
    g⁺::Array{Float64,1}
    p ::Array{Float64,1}
    s ::Array{Float64,1}
    y ::Array{Float64,1}
    H ::Array{Float64,2}
end

function BFGS_vars(n::Int64;sigma::Float64=1e-6)
    x = zeros(Float64,n);
    x⁺ = zeros(Float64,n);
    g = zeros(Float64,n);
    g⁺ = zeros(Float64,n);
    p = zeros(Float64,n);
    s = zeros(Float64,n);
    y = zeros(Float64,n);
    H = diagm(fill(sigma,n));
    return BFGS_vars(x,x⁺,g,g⁺,p,s,y,H)
end
    
function My_BFGS(func, grad!, x0, opts::BFGS_options, svars::BFGS_vars)
    itm = opts.itm;
    tol = opts.tol;
    ptf = opts.ptf;
    ifstats = opts.ifstats;
    stats = OptimizerStats(itm,ifstats);
    show_history = opts.show_history;
    
    n   = length(x0);
    x   = svars.x;  x⁺  = svars.x⁺;
    g   = svars.g;  g⁺  = svars.g⁺;
    p   = svars.p;  s   = svars.s;
    y   = svars.y;  H   = svars.H;

    d   = H[1,1];
    # re-new variables
    copy!(x, x0);
    copy!(x⁺, x);
    
    # initialization
    obj = func(x);
    grad!(g, x);
    # normalize!(g);
    # iterations
    noi = 0;
    err = vecnorm(g, Inf);

    updateOptimizerStats!(stats,obj,err,noi,ifstats)

    while err ≥ tol
        # p = -H⋅g
        BLAS.gemv!('N',-1.0,H,g,0.0,p);
        # line search with direction p
        flag, α = exact_BFGS!(x⁺, g⁺, x, g, p, grad!,H);
        # s = α⋅p, y = g⁺ - g
        for i = 1:n
            s[i] = α*p[i];
            y[i] = g⁺[i] - g[i];
        end
        
        # ρ = 1/yᵀs;
        ρ = 1.0/dot(y,s);
        μ = sum(abs2, s);
        ρ = min(10.0, ρ*μ)/μ;
        if ρ > 0.0
            # p = H⋅y
            BLAS.gemv!('N',1.0,H,y,0.0,p);
            # β = yᵀp⋅ρ² + ρ
            β = dot(y,p)*ρ^2 + ρ;
            # H ⟵ H - ρ(p⋅sᵀ + s⋅pᵀ) + β s⋅sᵀ
            # for j = 1:n, i = 1:n
            #     H[i,j] += -ρ*(p[i]*s[j] + s[i]*p[j]) + β*s[i]*s[j];
            # end
            BLAS.ger!(-ρ,s,p,H);
            BLAS.ger!(-ρ,p,s,H);
            BLAS.ger!( β,s,s,H);
        end
        # update history
        copy!(x, x⁺);
        copy!(g, g⁺);

        obj = func(x);
        err = vecnorm(g, Inf);
        noi += 1;
        updateOptimizerStats!(stats,obj,err,noi,ifstats)        
        (show_history && noi % ptf == 0) && @printf("BFGS: iter %3d, obj %1.5e, err %1.5e\n",
            noi, obj, err);
        # show_history && @show(ρ);
        if noi ≥ itm
            show_history && println("BFGS reach maximum iteration!");
            break;
        end
    end
    if ~opts.warm_start
        fill!(H,0.0);
        for i = 1:n
            H[i,i] = d;
        end
    end
end


#-----------------------------------------------------------------------------------------
# Exact Line Search
# goal: try to find a step size αᵏ ∈ ℝ₊₊ that satisfy
#       αᵏ = argmin f(xᵏ + α⋅pᵏ)
# idea: for solving the above optimization problem, it is equivalent to solve,
#       ⟨∇f(xᵏ + α⋅pᵏ), pᵏ⟩ = 0 (when function is strictly convex)
#       here we use linear interpolation to complish this job
# input:
#   x : current point
#   g : gradient at current point, g = ∇f(x)
#   p : descent direction
#   f : function handle
#   αmin: lower bound for α
#   tol : tolerence for bisection
# output:
#   x : inplace modified in the descent direction
#   g : inplace modified on the new point
#   flag 0: find x⁺ before α reach tol
#   flag 1: α reach tol
#   flag 2: reach the maximum stepsize 1.0
#----------------------------------------------------------------------------------------
function exact_BFGS!(x⁺ , g⁺, x, g, p, ∇f, H; αmin = 1e-10, tol = 1e-12)
    n   = length(x)
    l   = 0.0
    ml  = dot(g,p)

    if ml > 0.0
        println("Not a descent direction, restart BFGS...")
        # set p = -g
        copy!(p, g); scale!(p, -1.0)
        # set H = I
        fill!(H,0.0)
        for i = 1:n
            H[i,i] = 1.0
        end
    end

    # initialization and find the upper bound for α
    u   = 1.0;
    for i = 1:n
        x⁺[i] = x[i] + p[i]
    end
    ∇f(g⁺,x⁺)
    mu   = dot(g⁺, p)
    mu ≤ 0.0 && (return 2, 1.0)
    
    α = u
    m = mu
    noi = 0
    # start find root
    while abs(m) ≥ tol
        α = (l*abs(mu)+u*abs(ml))/(abs(ml)+abs(mu))
        α < αmin && (return 1, α)
        for i = 1:n
            x⁺[i] = x[i] + α*p[i]
        end
        ∇f(g⁺,x⁺)
        m = dot(g⁺,p)
        m > 0.0 ? (u = α; mu = m;) : (l = α; ml = m;)
        noi += 1
        noi >= 10 && break
    end
    # @show noi
    return 0, α
end
