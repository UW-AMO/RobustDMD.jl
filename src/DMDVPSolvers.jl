
############################################################
#
# BFGS Solvers
#
############################################################

###########################################################
# projection function w.r.t. b
function projb_BFGS!(vars, params, svars;
                     updatephi = false)
    updatephi && updatephimat!(vars.phi, params.t, vars.alpha);
    for id = 1:params.n
        projb_BFGS_sub!(vars,params,svars,id)
    end
    svars.novps = svars.novps + params.n
end

function projb_BFGS_subset!(vars, params, svars,
                            inds::Array{Int64,1}; updatephi=false)
    updatephi && updatephimat!(vars.phi, params.t, vars.alpha);    
    for i = 1:length(inds)
        id = inds[i]
        projb_BFGS_sub!(vars,params,svars,id)
    end
    svars.novps = svars.novps + length(inds)
end

###########################################################
# projection function w.r.t. bj
function projb_BFGS_sub!(vars, params, svars, id;
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
#
# Least Squares Solvers
#
############################################################

###########################################################
# projection function w.r.t. b
function projb_LSQ!(vars, params, svars;
    updatephi = false)
    updatephi && updatephimat!(vars.phi, params.t, vars.alpha);
    phitemp = svars.VPvars.phi
    copy!(phitemp,vars.phi)
    epsmin = svars.VPopts.epsmin
    DMD_LSQ_solve!(vars.B,phitemp,params.X,epsmin)
    svars.novps = svars.novps + params.n
end

function projb_LSQ_subset!(vars, params, svars,
    inds::Array{Int64,1}; updatephi=false)
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

###########################################################
# projection function w.r.t. bj
function projb_LSQ_sub!(vars, params, svars, id;
    updatephi=false)
    inds = [id]
    projb_LSQ_subset!(vars,params,svars,inds,
    updatephi=updatephi)
end

############################################################
# helper solver for LSQ (does all of the work)
function DMD_LSQ_solve!(B,phi,X,epsmin)

    c0 = zero(Complex{Float64});
    c1 = one(Complex{Float64});

    m, n = size(X)

    # stabilized least squares solution

    F = svdfact!(phi,thin=true)

    s1 = maximum(F[:S])
    k2 = sum(F[:S] .> s1*epsmin)

    Y = zeros(Complex{Float64},k2,n)
    U = view(F[:U],:,1:k2)
    Vt = view(F[:Vt],1:k2,:)
    BLAS.gemm!('C','N',c1,U,X,c0,Y)
    scale!(1./F[:S][1:k2],Y)
    BLAS.gemm!('C','N',c1,Vt,Y,c0,B)

end


############################################################
#
# Null Solvers (don't do anything for inner 
# solve, used for testing...)
#
############################################################

function projb_null!(vars, params, svars; updatephi=false)
    updatephi && updatephimat!(vars.phi, params.t, vars.alpha);    
    return
end

function projb_null_subset!(vars, params, svars, inds;
    updatephi=false)
    updatephi && updatephimat!(vars.phi, params.t, vars.alpha);        
    return
end

function projb_null_sub!(vars, params, svars, id;
    updatephi=false)
    updatephi && updatephimat!(vars.phi, params.t, vars.alpha);        
    return
end

