#==========================================================
    DMD Data Type
==========================================================#

##### parameters for DMD data #####
type DMDParams
    # dimensions
    m::Int64
    n::Int64
    k::Int64
    h::Int64
    # data
    X::Array{Complex{Float64},2}
    x::Array{Array{Complex{Float64},1},1}
    t::Array{Complex{Float64},1}
    # loss functions
    lossf::Function     # function
    lossg::Function     # gradient
end
# construct function
function DMDParams(k, h, X, t, lossf, lossg)
    m,n = size(X);
    x   = Array{Array{Complex{Float64},1}}(n);
    p   = pointer(X);
    s   = sizeof(Complex{Float64});
    for j = 1:n
        x[j] = unsafe_wrap(Array, p, m);
        p += m*s;
    end
    return DMDParams(m, n, k, h, X, x, t, lossf, lossg)
end

##### variables for DMD #####
type DMDVariables
    # decision variables
    alpha::Array{Complex{Float64},1}
    B::Array{Complex{Float64},2}
    b::Array{Array{Complex{Float64},1},1}
    phi::Array{Complex{Float64},2}
    # residual
    R::Array{Complex{Float64},2}
    r::Array{Array{Complex{Float64},1},1}
    # real correspondent of alpha and B
    alphar::Array{Float64,1}
    br::Array{Array{Float64,1},1}
end
# construct function
function DMDVariables(alpha, B, params::DMDParams)
    # complex 0 and 1
    c0 = zero(Complex{Float64});
    c1 = one(Complex{Float64});
    # renew alpha and B
    alpha0 = copy(alpha);
    B0 = copy(B);
    # dimensions
    m  = params.m;
    n  = params.n;
    k  = params.k;
    t  = params.t;
    X  = params.X;
    # allocate alphar
    pc = pointer(alpha0);
    pr = convert(Ptr{Float64}, pc);
    alphar = unsafe_wrap(Array, pr, 2*k);
    # allocate b and br
    b  = Array{Array{Complex{Float64},1}}(n);
    br = Array{Array{Float64,1}}(n);
    pc = pointer(B0);
    pr = convert(Ptr{Float64}, pc);
    sc = sizeof(Complex{Float64});
    sr = sizeof(Float64);
    for j = 1:n
        b[j]  = unsafe_wrap(Array, pc, k);
        br[j] = unsafe_wrap(Array, pr, 2*k);
        pc+= sc*k;
        pr+= sr*2*k;
    end
    # update phi
    phi  = zeros(Complex{Float64},m,k);
    updatephipsi!(phi, t, alpha0, exp);
    # update R = phi⋅B - X and r
    R  = copy(X);
    BLAS.gemm!('N','N',c1,phi,B0,-c1,R);
    r  = Array{Array{Complex{Float64},1}}(n);
    pc = pointer(R);
    for j = 1:n
        r[j] = unsafe_wrap(Array, pc, m);
        pc += m*sc;
    end

    return DMDVariables(alpha0, B0, b, phi, R, r, alphar, br)
end

##### Solver Variables #####

type DMDLSQ_options
    epsmin::Float64
end

type DMDLSQ_vars
    phi::Array{Complex{Float64},2}
end

function DMDLSQ_options( ; epsmin::Float64=1e-12)
    return DMDLSQ_options(epsmin)
end

function DMDLSQ_vars(params::DMDParams)
    m = params.m
    k = params.k
    phi = zeros(Complex{Float64},m,k)
    return DMDLSQ_vars(phi)
end

type DMDnull_options
    
end

type DMDnull_vars

end


type DMDVPSolverVariables
    VPvars::Union{BFGS_vars,DMDLSQ_vars,DMDnull_vars}
    VPopts::Union{BFGS_options,DMDLSQ_options,DMDnull_options}
    VPSolver!::Function
    VPSolver_sub!::Function
    VPSolver_subset!::Function
    novps::Int64
end

function DMDVPSolverVariablesBFGS(params::DMDParams;itm::Int64=500,
                                  tol::Float64=1e-7,ptf::Int64=100,
                                  show_history::Bool=false,
                                  warm_start::Bool=true,
                                  sigma::Float64 = 1e-6,
                                  ifstats::Bool = false)
    k = params.k
    VPopts = BFGS_options(itm,tol,ifstats,warm_start,show_history,ptf)
    d = k << 1;
    # b BFGS variables
    b   = zeros(d); b⁺  = zeros(d);
    gb  = zeros(d); gb⁺ = zeros(d);
    pb  = zeros(d); sb  = zeros(d);
    yb  = zeros(d); Hb  = diagm(fill(sigma, d));
    VPvars = BFGS_vars(b, b⁺, gb, gb⁺, pb, sb, yb, Hb);
    VPSolver! = (vars,params,svars;updatephi=false) -> projb_BFGS!(vars,
                                         params,svars;updatephi=updatephi)
    VPSolver_sub! = 
        (vars,params,svars,id;updatephi=false) -> projb_BFGS_sub!(vars,params,
                                               svars,id,updatephi=updatephi)
    VPSolver_subset! = 
        (vars,params,svars,inds;updatephi=false) -> projb_BFGS_subset!(vars,
                                               params,svars,inds;
                                               updatephi=updatephi)

    novps = 0
    return DMDVPSolverVariables(VPvars,VPopts,VPSolver!,VPSolver_sub!,
                                VPSolver_subset!,novps)

end

function DMDVPSolverVariablesLSQ(params::DMDParams;epsmin=1.0e-12)
    VPvars = DMDLSQ_vars(params);
    VPopts = DMDLSQ_options(epsmin);
    VPSolver! = (vars,params,svars;updatephi=false) -> projb_LSQ!(vars,
                                      params,svars,updatephi=updatephi)
    VPSolver_sub! = 
        (vars,params,svars,id;updatephi=false) -> projb_LSQ_sub!(vars,params,
                                               svars,id,updatephi=updatephi)
    VPSolver_subset! = 
        (vars,params,svars,inds;updatephi=false) -> projb_LSQ_subset!(vars,
                                    params,svars,inds,updatephi=updatephi)

    novps = 0
    return DMDVPSolverVariables(VPvars,VPopts,VPSolver!,VPSolver_sub!,
                                VPSolver_subset!,novps)

end

function DMDVPSolverVariablesNull(params)
    VPvars = DMDnull_vars();
    VPopts = DMDnull_options();
    VPSolver! = (vars,params,svars;updatephi=false) -> projb_null!(vars,
                                         params,svars;updatephi=updatephi)
    VPSolver_sub! = 
        (vars,params,svars,id;updatephi=false) -> projb_null_sub!(vars,params,
                                               svars,id,updatephi=updatephi)
    VPSolver_subset! = 
        (vars,params,svars,inds;updatephi=false) -> projb_null_subset!(vars,
                                               params,svars,inds;
                                               updatephi=updatephi)
    novps = 0
    return DMDVPSolverVariables(VPvars,VPopts,VPSolver!,VPSolver_sub!,
                                VPSolver_subset!,novps)

end

