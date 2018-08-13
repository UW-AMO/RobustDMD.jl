# This file is available under the terms of the MIT License

@doc """
Some common loss functions and their corresponding 
gradients
"""

###########################################################
# l2 penalty
function l2_func{T<:Union{AbstractFloat,Complex}}(Rc::Array{T})
    val = real(T(0.0));
    for I in eachindex(Rc)
        val += abs(Rc[I])^2;
    end
    return real(T(0.5))*val
end

function l2_grad!{T<:Union{AbstractFloat,Complex}}(Rc::Array{T})
    for I in eachindex(Rc)
        Rc[I] = real(T(0.5))*conj(Rc[I]);
    end
end

###########################################################
# Huber penalty
function huber_func{T<:Union{AbstractFloat,Complex}}(Rc::Array{T}, kappa::Real)
    val = real(T(0.0));
    rtkap = real(T(kappa));
    for I in eachindex(Rc)
        rho = abs(Rc[I]);
        rho ≤ rtkap ? val += real(T(0.5))*rho^2 : val += rtkap*rho - real(T(0.5))*rtkap^2
    end
    return val
end

function huber_grad!{T<:Union{AbstractFloat,Complex}}(Rc::Array{T}, kappa::Real)
    rtkap = real(T(kappa))
    for I in eachindex(Rc)
        rho = abs(Rc[I]);
        rho ≤ rtkap ? Rc[I] = real(T(0.5))*conj(Rc[I]) : Rc[I] = real(T(0.5))*rtkap*conj(Rc[I])/rho;
    end
end

###########################################################
# Student's T penalty
function st_func{T<:Union{AbstractFloat,Complex}}(Rc::Array{T}, nu::Real)
    val = real(T(0.0));
    rtnu = real(T(nu))
    for I in eachindex(Rc)
        rho = abs(Rc[I]);
        val += log(real(T(1.0)) + rho^2/rtnu);
    end
    return val
end

function st_grad!{T<:Union{AbstractFloat,Complex}}(Rc::Array{T}, nu::Real)
    rtnu = real(T(nu))
    for I in eachindex(Rc)
        rho = abs(Rc[I]);
        Rc[I] = conj(Rc[I])/(rtnu + rho^2)
    end
end
