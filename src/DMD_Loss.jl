###########################################################
# l2 penalty
function l2_func(r)
    T = real(eltype(r))
    return sum(abs2,r)*T(0.5)
end

function l2_grad!(r)
    conj!(r)
    T = eltype(r)
    BLAS.scal!(length(r),T(0.5),r,1)
end

###########################################################
# Huber penalty
function huber_func(r,kappa)
    T = real(eltype(r))
    val = T(0.0)
    Thalf = T(0.5)
    @inbounds for i in eachindex(r)
        ri = r[i];
        rho = abs(ri);
        val += rho <= kappa ? Thalf*rho^2 : kappa*rho - Thalf*kappa^2
    end
    return val
end

function huber_grad!(r,kappa)
    T = real(eltype(r))
    Thalf = T(0.5)
    Thalfkappa = Thalf*kappa
    @inbounds for i in eachindex(r)
        ri = r[i]
        rho = abs(ri);
        r[i] = rho <= kappa ? Thalf*conj(ri) : (Thalfkappa/rho)*conj(ri);
    end
end
