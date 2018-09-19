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
