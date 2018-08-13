# This file is available under the terms of the MIT License

@doc """
OptimizerStats type

stores various facts about the progress of the optimization

objs = value of objective function at each iteration
errs = maximum absolute value of objective gradient
noi = number of iterations
"""
mutable struct OptimizerStats{T}
    objs :: Array{T}
    errs :: Array{T}
    noi :: Integer
    maxiter :: Integer
end
    
function OptimizerStats(maxiter::Integer,ifstats::Bool,T::Type=Float64)
    if ifstats
        objs = Array{T}(maxiter+1)
        errs = Array{T}(maxiter+1)
        noi = 0
        return OptimizerStats(objs,errs,noi,maxiter)
    else
        return OptimizerStatsEmpty()
    end
end

function OptimizerStatsEmpty()
    objs = []
    errs = []
    noi = 0
    maxiter = 0
    return OptimizerStats(objs,errs,noi,maxiter)
end

function updateOptimizerStats!{T<:AbstractFloat}(stats::OptimizerStats{T},
                               obj,err,
                               noi::Integer,ifstats::Bool)

    if (ifstats && noi <= stats.maxiter && noi >= 0)
        stats.objs[noi+1] = T(obj)
        stats.errs[noi+1] = T(err)
        stats.noi = noi
    end

    return
end
    
function updateOptimizerStats!(stats::OptimizerStats{Any},
                               obj,err,
                               noi::Integer,ifstats::Bool)

    if (ifstats && noi <= stats.maxiter && noi >= 0)
        stats.objs[noi+1] = obj
        stats.errs[noi+1] = err
        stats.noi = noi
    end

    return
end
    
