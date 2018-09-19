# This file is available under the terms of the MIT License

@doc """
Functions for evaluating the objective and gradient.
These are mostly wrappers of the tools in DMDUtil.jl.
"""

###########################################################
# DMD objective function 
#   make sure you already update all alpha, B related vars
function objective(ar, vars, params, svars;
    updateP=true, updateB=true, updateR=true)
    
    copy!(vars.ar, ar);
    # update phi
    updateP && (update_P!(vars, params);
        update_PQR!(vars, params, svars));
    # update B
    updateB && update_B!(vars, params, svars);
    # update Residual
    updateR && update_R!(vars, params);
    
    # calculate objective value
    return 0.5*sum(abs2, vars.R);
end

function objective(ar, vars, params, svars, id;
    updateP=true, updateB=true, updateR=true)

    copy!(vars.ar, ar);
    # update phi
    updateP && (update_P!(vars, params);
        update_PQR!(vars, params, svars));
    # update B
    updateB && update_b!(vars, params, svars, id);
    # update Residual
    updateR && update_r!(vars, params, id);
    
    # calculate objective value
    return 0.5*sum(abs2, vars.r[id]);
end

###########################################################
# gradient w.r.t. alpha
function gradient!(ar, gar, vars, params, svars;
    updateP=true, updateB=true, updateR=true)
    
    copy!(vars.ar, ar);
    # update phi
    updateP && (update_P!(vars, params);
        update_PQR!(vars, params, svars));
    # update B
    updateB && update_B!(vars, params, svars);
    # update Residual
    updateR && update_R!(vars, params);
    # calculate gradient
    grad_ar!(gar, vars, params, svars);
end

function gradient!(ar, gar, vars, params, svars, id;
    updateP=true, updateB=true, updateR=true)
    
    copy!(vars.ar, ar);
    # update phi
    updateP && (update_P!(vars, params);
        update_PQR!(vars, params, svars));
    # update B
    updateB && update_b!(vars, params, svars, id);
    # update Residual
    updateR && update_r!(vars, params, id);
    # calculate gradient
    grad_ar!(gar, vars, params, svars, id);
end

# ###########################################################
# # function value and gradient w.r.t. alpha
# function alpha_fg!(alphar, galphar, vars, params, svars)
#     VPSolver! = svars.VPSolver!
#     # inner solve
#     copy!(vars.alphar, alphar);
#     updatephimat!(vars.phi, params.t, vars.alpha);
#     VPSolver!(vars,params,svars);
#     # calculate function value
#     func = DMDObj(vars, params, updateR = true);
#     # calculate gradient
#     dmd_alphagrad1!(galphar,vars,params);

#     return func
# end

# ###########################################################
# # objective w.r.t. b[id]
# function bfunc_sub(br, vars, params, id)
#     # update b related variables
#     copy!(vars.br[id], br);
#     # return function value
#     return DMDObj_sub(vars, params, id, updateR = true)
# end

# ###########################################################
# # gradient w.r.t. b[id]
# function bgrad_sub!{T<:Union{Float64,Float32}}(br::Array{T}, gbr::Array{T}, vars::DMDVariables{T}, params::DMDParams{T}, id)
#     # update b related variables
#     copy!(vars.br[id], br);
#     # update Residual
#     updateResidual_sub!(vars, params, id);
#     params.lossg(vars.r[id]);
#     # wrap complex array around gbr
#     pr = pointer(gbr);
#     pc = convert(Ptr{Complex{T}}, pr);
#     gb = unsafe_wrap(Array, pc, params.k);
#     # obtain complex gradient
#     c0 = zero(Complex{T});
#     c1 = one(Complex{T});
#     BLAS.gemm!('T','N',c1,vars.phi,vars.r[id],c0,gb);
#     # transfer to real gradient
#     scale!(gbr, T(-2.0));
#     BLAS.scal!(params.k,T(-1.0),gbr,2);
# end

# ###########################################################
# # gradient w.r.t. alpha for the jth function
# function alphagrad_sub!{T<:Union{Float64,Float32}}(galphar::Array{T}, vars::DMDVariables{T}, params::DMDParams{T}, id)
#     # calculate gradient
#     updateResidual_sub!(vars, params, id);
#     params.lossg(vars.r[id]);
#     # wrap complex array around galphar
#     pr = pointer(galphar);
#     pc = convert(Ptr{Complex{T}}, pr);
#     galpha = unsafe_wrap(Array, pc, params.k);
#     # obtain complex gradient
#     # TODO: This step need to be optimized
#     # BLAS.sum!(galpha, (vars.phi.'*diagm(params.t)*vars.R).*vars.B);
#     broadcast!(*,vars.r[id],vars.r[id],params.t);
#     c0 = zero(Complex{T});
#     c1 = one(Complex{T});
#     BLAS.gemv!('T',c1,vars.phi,vars.r[id],c0,galpha);
#     broadcast!(*,galpha,galpha,vars.b[id]);
#     # transfer to real gradient
#     scale!(galphar, T(-2.0));
#     BLAS.scal!(params.k,T(-1.0),galphar,2);
# end
