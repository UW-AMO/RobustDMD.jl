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

mutable struct BFGS_options{T<:AbstractFloat}
    itm::Integer
    tol::T
    ifstats::Bool
    warm_start::Bool    
    show_history::Bool
    ptf::Integer
end

mutable struct BFGS_vars{T<:AbstractFloat}
    x ::Array{T,1}
    x⁺::Array{T,1}
    g ::Array{T,1}
    g⁺::Array{T,1}
    p ::Array{T,1}
    s ::Array{T,1}
    y ::Array{T,1}
    H ::Array{T,2}
end

function BFGS_vars{T<:AbstractFloat}(n::Integer,alpha::T,sigma::T = sqrt(eps(T)))
    x = zeros(T,n);
    x⁺ = zeros(T,n);
    g = zeros(T,n);
    g⁺ = zeros(T,n);
    p = zeros(T,n);
    s = zeros(T,n);
    y = zeros(T,n);
    H = diagm(fill(sigma,n));
    return BFGS_vars(x,x⁺,g,g⁺,p,s,y,H)
end

function BFGS_vars(n::Integer,T::Type=Float64)
    return BFGS_vars(n,one(T))
end
    
function My_BFGS{T<:AbstractFloat}(func, grad!, x0::Array{T}, opts::BFGS_options, svars::BFGS_vars{T})
    itm = opts.itm;
    tol = opts.tol;
    ptf = opts.ptf;
    ifstats = opts.ifstats;
    stats = OptimizerStats(itm,ifstats,T);
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
        BLAS.gemv!('N',T(-1.0),H,g,T(0.0),p);
        # line search with direction p
        flag, α = exact_BFGS!(x⁺, g⁺, x, g, p, grad!,H);
        #(flag == 1) && @show (flag, noi)
        #(err > 1e5) && @show noi
        # s = α⋅p, y = g⁺ - g
        for i = 1:n
            s[i] = α*p[i];
            y[i] = g⁺[i] - g[i];
        end
        
        # ρ = 1/yᵀs;
        ρ = T(1.0)/dot(y,s);
        μ = sum(abs2, s);
        ρ = min(T(10.0), ρ*μ)/μ;
        if ρ > 0.0
            # p = H⋅y
            BLAS.gemv!('N',T(1.0),H,y,T(0.0),p);
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
        fill!(H,T(0.0));
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
function exact_BFGS!{T<:AbstractFloat}(x⁺::Array{T} , g⁺, x, g, p, ∇f, H; αmin = T(1e1)*eps(T), tol = T(1e2)*eps(T))
    n   = length(x)
    l   = T(0.0)
    ml  = dot(g,p)

    #@show sum(isnan(p)), sum(isnan(H))
    if ml > 0.0
        #@show vecnorm(H)
        #@show vecnorm(g), vecnorm(p)
        #println("Not a descent direction, restart BFGS... ml = ",ml)
        # set p = -g
        copy!(p, g); scale!(p, T(-1.0e-5))
        # set H = I
        fill!(H,T(0.0))
        for i = 1:n
            H[i,i] = T(1.0e-5)
        end
        ml = dot(g,p)
    end

    # initialization and find the upper bound for α
    u   = T(1.0);
    for i = 1:n
        x⁺[i] = x[i] + p[i]
    end
    ∇f(g⁺,x⁺)
    mu   = dot(g⁺, p)
    mu ≤ 0.0 && (return 2, T(1.0))
    
    α = u
    m = mu
    noi = 0
    # start find root 
    while abs(m) ≥ tol
        α = (l*abs(mu)+u*abs(ml))/(abs(ml)+abs(mu))
        #α < αmin && (return 1, α)
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
