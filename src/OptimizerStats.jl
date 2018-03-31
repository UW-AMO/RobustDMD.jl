
#============================================================
OptimizerStats type

stores various facts about the progress of the optimization

objs = value of objective function at each iteration
errs = maximum absolute value of objective gradient
noi = number of iterations

============================================================#

type OptimizerStats
    objs :: Array{Float64,1}
    errs :: Array{Float64,1}
    noi :: Int64
    maxiter :: Int64
end
    
function OptimizerStats(maxiter::Integer,ifstats::Bool)
    if ifstats
        objs = Array(Float64,maxiter+1)
        errs = Array(Float64,maxiter+1)
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

function updateOptimizerStats!(stats::OptimizerStats,
                               obj::Float64,err::Float64,
                               noi::Integer,ifstats::Bool)

    if (ifstats && noi <= stats.maxiter && noi >= 0)
        stats.objs[noi+1] = obj
        stats.errs[noi+1] = err
        stats.noi = noi
    end

    return
end
    
