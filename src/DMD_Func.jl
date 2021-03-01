using Optim
export dmdobjective, projBl2

# This file is available under the terms of the MIT License

@doc """
Functions for evaluating the objective and gradient.
"""

function dmdobjective(params;updatep=true)
    if updatep
        return aBFunc(params)
    else
        return BFunc(params)
    end
end

function BFunc(params)
    T = eltype(params.X);
    # update residual
    BLAS.gemm!('N', 'N', T(1.0), params.P, params.B, T(0.0), params.R);
    params.R .-= params.X
    #
    return params.lossFunc(view(params.R, :, params.ikeep))
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

function bFuncGrad(gbr, params, id)
    T = eltype(params.X);
    Tr = typeof(real(params.X[1]));
    # update residual
    BLAS.gemv!('N', T(1.0), params.P, params.b[id], T(0.0), params.r[id]);
    params.r[id] .-= params.x[id];
    #

    
    f = params.lossFunc(params.r[id])
    
    params.lossGrad(params.r[id]);
    #
    gb = vr2vc(gbr);
    BLAS.gemv!('T', T(1.0), params.P, params.r[id], T(0.0), gb);
    gbr[1:2:end] .*= Tr(2.0);
    gbr[2:2:end] .*= Tr(-2.0);

    return f
end


function bfg!(ft, gbr, br, params, id)
    copyto!(params.br[id], br);
    if ft != nothing && gbr != nothing
        return bFuncGrad(gbr, params, id);
    end
    if ft == nothing && gbr != nothing
        bGrad(gbr, params, id)
        return
    end
    if ft != nothing && gbr == nothing
        return bFunc(params, id)
    end
end

function projb(params, id)
    
    # define the function and gradient interface for optim
    #

    function obfg!(f, g, x)
        return bfg!(f, g, x, params, id)
    end
    
    optimfg! = Optim.only_fg!(obfg!)
    res = optimize(optimfg!, params.br[id], params.inner_solver,
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
    if params.inner_directl2
        projBl2(params);
    else
        projB(params);
    end
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
    params.r[id] .*= params.t;
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

    if params.inner_directl2
        projBl2(params);
    else
        projB(params);
    end

    # trim
    trimstandard(params)
    
    # update residual
    BLAS.gemm!('N', 'N', T(1.0), params.P, params.B, T(0.0), params.R);
    params.R .-= params.X
    #
    params.lossGrad(params.R);
    #
    ga = vr2vc(gar);
    #
    params.R .*= params.t;
    BLAS.gemm!('T', 'N', T(1.0), params.P, params.R, T(0.0), params.tM);
    params.tM .*= params.B;
    sum!(ga, view(params.tM, :, params.ikeep));
    #
    gar[1:2:end] .*= Tr(2.0);
    gar[2:2:end] .*= Tr(-2.0);
end

function aBFuncGrad(gar, params)
    # update P
    update_P!(params);
    #
    T = eltype(params.X);
    Tr = typeof(real(params.X[1]));
    # partially minimize over B
    if params.inner_directl2
        projBl2(params);
    else
        projB(params);
    end

    # trim
    trimstandard(params)
    
    # update residual
    BLAS.gemm!('N', 'N', T(1.0), params.P, params.B, T(0.0), params.R);
    params.R .-= params.X
    #

    f = params.lossFunc(view(params.R, :, params.ikeep))    
    
    params.lossGrad(params.R);
    #
    ga = vr2vc(gar);
    #
    params.R .*= params.t;
    BLAS.gemm!('T', 'N', T(1.0), params.P, params.R, T(0.0), params.tM);
    params.tM .*= params.B;
    sum!(ga, view(params.tM, :, params.ikeep));
    #
    gar[1:2:end] .*= Tr(2.0);
    gar[2:2:end] .*= Tr(-2.0);

    return f
end


#####################################################
# L2 Penalty

function updateqr!(params)
    if params.a != params.qra
        qrfact = params.qrfact;

        # grab pointers
        fact = qrfact.factors
        jpvt = qrfact.jpvt
        tau = qrfact.τ

        # prep and call factorization
        copyto!(fact, params.P)
        jpvt .= 0 # do not forget!
        LinearAlgebra.LAPACK.geqp3!(fact, jpvt, tau)
        copyto!(params.qra, params.a)
    end
end


function projbl2(params, id)
    updateqr!(params)
    copyto!(params.qrtemp[id], params.x[id])
    ldiv!(params.qrfact, params.qrtemp[id])
    k = params.k
    copyto!(params.b[id], view(params.qrtemp[id], 1:k))
end

function projBl2(params)
    if any(isinf, params.P)
        params.B .= 0.0
    else
        updateqr!(params)
        copyto!(params.qrTemp, params.X)
        ldiv!(params.qrfact, params.qrTemp)
        k = params.k
        copyto!(params.B, view(params.qrTemp, 1:k, :))
    end
end

################################################
# trimming

function trimstandard(params)
    if params.n == params.nkeep
        return
    end
    T = eltype(params.X)
    # update residual
    BLAS.gemm!('N', 'N', T(1.0), params.P, params.B, T(0.0), params.R);
    params.R .-= params.X
    for id = 1:params.n
        params.tcnorm[id] = params.lossFunc(params.r[id])
    end
    sortperm!(params.idsort, params.tcnorm, rev=false)
    copyto!(params.ikeep, view(params.idsort, 1:params.nkeep))
    sort!(params.ikeep)
end    
