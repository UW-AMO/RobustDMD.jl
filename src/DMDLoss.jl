#==========================================================
    DMD Loss Functions and Corresponding Gradient
==========================================================#

###########################################################
# l2 penalty
function l2_func(Rc)
    val = 0.0;
    for I in eachindex(Rc)
        val += abs(Rc[I])^2;
    end
    return 0.5*val
end

function l2_grad!(Rc)
    for I in eachindex(Rc)
        Rc[I] = 0.5*conj(Rc[I]);
    end
end

###########################################################
# Huber penalty
function huber_func(Rc, kappa)
    val = 0.0;
    for I in eachindex(Rc)
        rho = abs(Rc[I]);
        rho ≤ kappa ? val += 0.5*rho^2 : val += kappa*rho - 0.5*kappa^2
    end
    return val
end

function huber_grad!(Rc, kappa)
    for I in eachindex(Rc)
        rho = abs(Rc[I]);
        rho ≤ kappa ? Rc[I] = 0.5*conj(Rc[I]) : Rc[I] = 0.5*kappa*conj(Rc[I])/rho;
    end
end

###########################################################
# Student's T penalty
function st_func(Rc, nu)
    val = 0.0;
    for I in eachindex(Rc)
        rho = abs(Rc[I]);
        val += log(1.0 + rho^2/nu);
    end
    return val
end

function st_grad!(Rc, nu)
    for I in eachindex(Rc)
        rho = abs(Rc[I]);
        Rc[I] = conj(Rc[I])/(nu + rho^2)
    end
end
