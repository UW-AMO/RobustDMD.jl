# This file is available under the terms of the MIT License

@doc """
Functions for evaluating the objective and gradient.
These are mostly wrappers of the tools in DMDUtil.jl.
"""

###########################################################
# DMD objective function 
# #   make sure you already update all alpha, B related vars
# function objective(ar, vars, params, svars;
#     updateP=true, updateB=true, updateR=true)
    
#     copy!(vars.ar, ar);
#     # update phi
#     updateP && (update_P!(vars, params);
#         update_PQR!(vars, params, svars));
#     # update B
#     updateB && update_B!(vars, params, svars);
#     # update Residual
#     updateR && update_R!(vars, params);
    
#     # calculate objective value
#     return 0.5*sum(abs2, vars.R);
# end

function BFunc(params)
    T = eltype(params.X);
    # update residual
    BLAS.gemm!('N', 'N', T(1.0), params.P, params.B, T(0.0), params.R);
    params.R .-= params.X
    #
    return params.lossFunc(params.R)
end

function bFunc(params, id)
    T = eltype(params.X);
    # update residual
    BLAS.gemv!('N', T(1.0), params.P, params.b[id], T(0.0), params.r[id]);
    params.r[id] .-= params.x[id]
    #
    return params.lossFunc(params.r[id])
end

function bGrad(gbr, params, id)
    T = eltype(params.X);
    Tr = typeof(real(params.X[1]));
    # update residual
    BLAS.gemv!('N', T(1.0), params.P, params.b[id], T(0.0), params.r[id]);
    params.r[id] .-= params.x[id];
    #
    params.lossGrad(params.r[id]);
    #
    gb = vr2vc(gbr);
    BLAS.gemv!('T', T(1.0), params.P, params.r[id], T(0.0), gb);
    gbr[1:2:end] .*= Tr(2.0);
    gbr[2:2:end] .*= Tr(-2.0);
end


function projbl2(params, id)
    copyto!(params.b[id],params.P\params.x[id])
end

function projb(params, id)
    # define the function and gradient interface for optim
    function f(br)
        copyto!(params.br[id], br);
        return bFunc(params, id)
    end
    #
    function g!(gbr, br)
        copyto!(params.br[id], br);
        bGrad(gbr, params, id);
    end
    
    res = optimize(f, g!, params.br[id], BFGS(),
                   params.inner_opts);
    
    copyto!(params.br[id], res.minimizer);

end

function projB(params)
    for id = 1:params.n
        projb(params, id);
    end
end

function abFunc(params, id)
    # update P
    update_P!(params);
    # partially minimize over b
    projb(params, id);
    #
    return bFunc(params, id)
end

function aBFunc(params)
    # update P
    update_P!(params);
    # partially minimize over B
    projB(params);
    #
    return BFunc(params)
end

function abGrad(gar, params, id)
    # update P
    update_P!(params);
    #
    T = eltype(params.X);
    Tr = typeof(real(params.X[1]));
    # partially minimize over b
    projb(params, id);
    # update residual
    BLAS.gemv!('N', T(1.0), params.P, params.b[id], T(0.0), params.r[id]);
    params.r[id] .-= params.x[id];
    #
    params.lossGrad(params.r[id]);
    #
    ga = vr2vc(gar);
    #
    params.r[id] .*= t;
    BLAS.gemv!('T', T(1.0), params.P, params.r[id], T(0.0), ga);
    ga .*= params.b[id];
    #
    gar[1:2:end] .*= Tr(2.0);
    gar[2:2:end] .*= Tr(-2.0);
end

function aBGrad(gar, params)
    # update P
    update_P!(params);
    #
    T = eltype(params.X);
    Tr = typeof(real(params.X[1]));
    # partially minimize over B
    projB(params);
    # update residual
    BLAS.gemm!('N', 'N', T(1.0), params.P, params.B, T(0.0), params.R);
    params.R .-= params.X
    #
    params.lossGrad(params.R);
    #
    ga = vr2vc(gar);
    #
    params.R .*= t;
    BLAS.gemm!('T', 'N', T(1.0), params.P, params.R, T(0.0), params.tM);
    params.tM .*= params.B;
    sum!(ga, params.tM);
    #
    gar[1:2:end] .*= Tr(2.0);
    gar[2:2:end] .*= Tr(-2.0);
end

###########################################################
# gradient w.r.t. alpha
# function gradient!(ar, gar, vars, params, svars;
#     updateP=true, updateB=true, updateR=true)
    
#     copy!(vars.ar, ar);
#     # update phi
#     updateP && (update_P!(vars, params);
#         update_PQR!(vars, params, svars));
#     # update B
#     updateB && update_B!(vars, params, svars);
#     # update Residual
#     updateR && update_R!(vars, params);
#     # calculate gradient
#     grad_ar!(gar, vars, params, svars);
# end

# function gradient!(ar, gar, vars, params, svars, id;
#     updateP=true, updateB=true, updateR=true)
    
#     copy!(vars.ar, ar);
#     # update phi
#     updateP && (update_P!(vars, params);
#         update_PQR!(vars, params, svars));
#     # update B
#     updateB && update_b!(vars, params, svars, id);
#     # update Residual
#     updateR && update_r!(vars, params, id);
#     # calculate gradient
#     grad_ar!(gar, vars, params, svars, id);
# end
