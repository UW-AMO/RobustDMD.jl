# This file is available under the terms of the MIT License

@doc """
Functions for evaluating the objective and gradient.
These are mostly wrappers of the tools in DMDUtil.jl.
"""

###########################################################
# DMD objective function 
#   make sure you already update all alpha, B related vars
function DMDObj(vars::DMDVariables, params::DMDParams;
    updatephi = false, updateR = false)

    # update phi
    updatephi && updatephimat!(vars.phi, params.t, vars.alpha);
    # update Residual
    updateR && updateResidual!(vars, params);
    if (any(isnan,vars.R) || any(isinf,vars.R))
        return real(eltype(vars.R))(Inf)
    end
    # calculate objective value
    return params.lossf(vars.R)
end

function DMDObj_sub(vars::DMDVariables, params::DMDParams, id;
    updatephi = false, updateR = false)
    # update phi
    updatephi && updatephimat!(vars.phi, params.t, vars.alpha);
    # update Residual
    updateR && updateResidual_sub!(vars, params, id);
    if (any(isnan,vars.r[id]) || any(isinf,vars.r[id]))
        return real(eltype(vars.R))(Inf)
    end
    return params.lossf(vars.r[id])
end

###########################################################
# objective w.r.t. alpha
function alphafunc(alphar, vars, params, svars)
    VPSolver! = svars.VPSolver!
    # update alpha related variables
    copy!(vars.alphar, alphar);
    updatephimat!(vars.phi, params.t, vars.alpha);
    if (any(isnan,vars.phi) || any(isinf,vars.phi))
        return real(eltype(vars.R))(Inf)
    end
    # variable projection, project b
    VPSolver!(vars,params,svars);
    # calculate objective value
    return DMDObj(vars, params, updateR = true)
end

###########################################################
# gradient w.r.t. alpha
function alphagrad!(alphar, galphar, vars, params, svars)
    VPSolver! = svars.VPSolver!
    # update alpha related variables
    copy!(vars.alphar, alphar);
    updatephimat!(vars.phi, params.t, vars.alpha);
    # variable projection, project b
    VPSolver!(vars,params,svars);
    # update residual (new b)
    updateResidual!(vars, params)
    # calculate gradient
    dmd_alphagrad1!(galphar,vars,params)

end

###########################################################
# function value and gradient w.r.t. alpha
function alpha_fg!(alphar, galphar, vars, params, svars)
    VPSolver! = svars.VPSolver!
    # inner solve
    copy!(vars.alphar, alphar);
    updatephimat!(vars.phi, params.t, vars.alpha);
    VPSolver!(vars,params,svars);
    # calculate function value
    func = DMDObj(vars, params, updateR = true);
    # calculate gradient
    dmd_alphagrad1!(galphar,vars,params);

    return func
end

###########################################################
# objective w.r.t. b[id]
function bfunc_sub(br, vars, params, id)
    # update b related variables
    copy!(vars.br[id], br);
    # return function value
    return DMDObj_sub(vars, params, id, updateR = true)
end

###########################################################
# gradient w.r.t. b[id]
function bgrad_sub!{T<:Union{Float64,Float32}}(br::Array{T}, gbr::Array{T}, vars::DMDVariables{T}, params::DMDParams{T}, id)
    # update b related variables
    copy!(vars.br[id], br);
    # update Residual
    updateResidual_sub!(vars, params, id);
    params.lossg(vars.r[id]);
    # wrap complex array around gbr
    pr = pointer(gbr);
    pc = convert(Ptr{Complex{T}}, pr);
    gb = unsafe_wrap(Array, pc, params.k);
    # obtain complex gradient
    c0 = zero(Complex{T});
    c1 = one(Complex{T});
    BLAS.gemm!('T','N',c1,vars.phi,vars.r[id],c0,gb);
    # transfer to real gradient
    scale!(gbr, T(-2.0));
    BLAS.scal!(params.k,T(-1.0),gbr,2);
end

###########################################################
# gradient w.r.t. alpha for the jth function
function alphagrad_sub!{T<:Union{Float64,Float32}}(galphar::Array{T}, vars::DMDVariables{T}, params::DMDParams{T}, id)
    # calculate gradient
    updateResidual_sub!(vars, params, id);
    params.lossg(vars.r[id]);
    # wrap complex array around galphar
    pr = pointer(galphar);
    pc = convert(Ptr{Complex{T}}, pr);
    galpha = unsafe_wrap(Array, pc, params.k);
    # obtain complex gradient
    # TODO: This step need to be optimized
    # BLAS.sum!(galpha, (vars.phi.'*diagm(params.t)*vars.R).*vars.B);
    broadcast!(*,vars.r[id],vars.r[id],params.t);
    c0 = zero(Complex{T});
    c1 = one(Complex{T});
    BLAS.gemv!('T',c1,vars.phi,vars.r[id],c0,galpha);
    broadcast!(*,galpha,galpha,vars.b[id]);
    # transfer to real gradient
    scale!(galphar, T(-2.0));
    BLAS.scal!(params.k,T(-1.0),galphar,2);
end
