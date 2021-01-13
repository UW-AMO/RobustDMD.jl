using Optim
export solveDMD_withOptimBFGS


# BFGS solver
function solveDMD_withOptimBFGS(params;opts=Optim.Options())

    function fg!(f,gr,ar)
        if f != nothing && gr != nothing
            copyto!(params.ar,ar)
            return aBFuncGrad(gr,params)
        end
        if f == nothing && gr != nothing
            copyto!(params.ar,ar)
            aBGrad(gr,params)
            return
        end
        if f != nothing && gr == nothing
            copyto!(params.ar,ar)
            return aBFunc(params)
        end
        return
    end

    res = optimize(Optim.only_fg!(fg!),copy(params.ar),Optim.BFGS())
    return res
    
end
