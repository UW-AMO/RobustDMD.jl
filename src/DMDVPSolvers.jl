# This file is available under the terms of the MIT License

@doc """
Built-in tools for inner solves. Includes the DMDVPSolverVariables
struct, which can be used to define a custom inner solver
"""

############################################################
# DMDVPSolverVariables struct

mutable struct DMDVPSolverVariables{A,B}
    VPvars::A
    VPopts::B
    VPSolver!::Function
    VPSolver_sub!::Function
    VPSolver_subset!::Function
    novps::Integer
end


############################################################
# BFGS Solvers


# projection function w.r.t. b
function projb_BFGS!(vars::DMDVariables, params::DMDParams, svars::DMDVPSolverVariables;
                     updatephi = false)
    updatephi && updatephimat!(vars.phi, params.t, vars.alpha);
    Threads.@threads for id = 1:params.n
        #@show id
        projb_BFGS_sub!(vars,params,svars,id)
    end
    svars.novps = svars.novps + params.n
end

function projb_BFGS_subset!{T<:Integer}(vars::DMDVariables, params::DMDParams, svars::DMDVPSolverVariables,
                            inds::Array{T}; updatephi=false)
    updatephi && updatephimat!(vars.phi, params.t, vars.alpha);    
    Threads.@threads for i = 1:length(inds)
        id = inds[i]
        projb_BFGS_sub!(vars,params,svars,id)
    end
    svars.novps = svars.novps + length(inds)
end

# projection function w.r.t. bj
function projb_BFGS_sub!(vars::DMDVariables, params::DMDParams, svars::DMDVPSolverVariables, id;
                         updatephi=false)
    updatephi && updatephimat!(vars.phi, params.t, vars.alpha);
    VPvars = svars.VPvars
    VPopts = svars.VPopts
    func = (br) -> bfunc_sub(br, vars, params, id);
    grad! = (gbr,br) -> bgrad_sub!(br, gbr, vars, params, id);
    my_res = My_BFGS(func, grad!, vars.br[id], VPopts, VPvars);
    svars.novps = svars.novps + 1
end

############################################################
# Least Squares Solvers

# projection function w.r.t. b
function projb_LSQ!(vars::DMDVariables, params::DMDParams, svars::DMDVPSolverVariables;
    updatephi = false)
    updatephi && updatephimat!(vars.phi, params.t, vars.alpha);
    phitemp = svars.VPvars.phi
    copy!(phitemp,vars.phi)
    epsmin = svars.VPopts.epsmin
    DMD_LSQ_solve!(vars.B,phitemp,params.X,epsmin)
    svars.novps = svars.novps + params.n
end

function projb_LSQ_subset!{T<:Integer}(vars::DMDVariables, params::DMDParams, svars::DMDVPSolverVariables,
    inds::Array{T}; updatephi=false)
#    projb_LSQ!(vars, params, svars, updatephi = updatephi)
    updatephi && updatephimat!(vars.phi, params.t, vars.alpha);    
    phitemp = svars.VPvars.phi
    copy!(phitemp,vars.phi)
    epsmin = svars.VPopts.epsmin
    Xtemp = copy(params.X[:,inds])
    Btemp = copy(vars.B[:,inds])
    DMD_LSQ_solve!(Btemp,phitemp,Xtemp,epsmin)
    vars.B[:,inds] = Btemp
    svars.novps = svars.novps + length(inds)
end

# projection function w.r.t. bj
function projb_LSQ_sub!(vars::DMDVariables, params::DMDParams, svars::DMDVPSolverVariables, id;
    updatephi=false)
    inds = [id]
    projb_LSQ_subset!(vars,params,svars,inds,
    updatephi=updatephi)
end

# helper solver for LSQ (does all of the work)
function DMD_LSQ_solve!{T<:Union{Float64,Float32}}(B::Array{Complex{T}},phi::Array{Complex{T}},X::Array{Complex{T}},epsmin::T)

    c0 = zero(Complex{T});
    c1 = one(Complex{T});

    m, n = size(X)

    if (any(isnan,phi) || any(isinf,phi))
        # error handling (probably a bad alpha...)
        fill!(B,T(0.0))
    else
        # stabilized least squares solution

        F = svdfact(phi,thin=true)

        s1 = maximum(F[:S])
        k2 = sum(F[:S] .> s1*epsmin)

        Y = zeros(Complex{T},k2,n)
        U = view(F[:U],:,1:k2)
        Vt = view(F[:Vt],1:k2,:)
        BLAS.gemm!('C','N',c1,U,X,c0,Y)
        scale!(1./F[:S][1:k2],Y)
        BLAS.gemm!('C','N',c1,Vt,Y,c0,B)
    end
end


############################################################
# Null Solvers (don't do anything for inner 
# solve, used for testing...)

function projb_null!(vars::DMDVariables, params::DMDParams, svars::DMDVPSolverVariables; updatephi=false)
    updatephi && updatephimat!(vars.phi, params.t, vars.alpha);    
    return
end

function projb_null_subset!(vars::DMDVariables, params::DMDParams, svars::DMDVPSolverVariables, inds;
    updatephi=false)
    updatephi && updatephimat!(vars.phi, params.t, vars.alpha);        
    return
end

function projb_null_sub!(vars::DMDVariables, params::DMDParams, svars::DMDVPSolverVariables, id;
    updatephi=false)
    updatephi && updatephimat!(vars.phi, params.t, vars.alpha);        
    return
end

##################################################
# Solver Variables

mutable struct DMDLSQ_options{T<:AbstractFloat}
    epsmin::T
end

mutable struct DMDLSQ_vars{T<:AbstractFloat}
    phi::Array{Complex{T}}
end

function DMDLSQ_options{T<:AbstractFloat}( ; epsmin::T=T(1e2)*eps(T))
    return DMDLSQ_options(epsmin)
end

function DMDLSQ_vars{T<:AbstractFloat}(params::DMDParams{T})
    m = params.m
    k = params.k
    phi = zeros(Complex{T},m,k)
    return DMDLSQ_vars(phi)
end

mutable struct DMDnull_options
    
end

mutable struct DMDnull_vars

end


function DMDVPSolverVariablesBFGS{T<:AbstractFloat}(params::DMDParams{T};itm::Integer=500,
                                  tol::T=sqrt(eps(T)),ptf::Integer=100,
                                  show_history::Bool=false,
                                  warm_start::Bool=true,
                                  sigma::T = sqrt(eps(T)),
                                  ifstats::Bool = false)
    k = params.k
    VPopts = BFGS_options(itm,tol,ifstats,warm_start,show_history,ptf)
    d = k << 1;
    # b BFGS variables
    b   = zeros(T,d); b⁺  = zeros(T,d);
    gb  = zeros(T,d); gb⁺ = zeros(T,d);
    pb  = zeros(T,d); sb  = zeros(T,d);
    yb  = zeros(T,d); Hb  = diagm(fill(sigma, d));
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

function DMDVPSolverVariablesLSQ{T<:AbstractFloat}(params::DMDParams{T};epsmin=T(1e2)*eps(T))
    VPvars = DMDLSQ_vars(params);
    VPopts = DMDLSQ_options(T(epsmin));
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

function DMDVPSolverVariablesNull(params::DMDParams)
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

