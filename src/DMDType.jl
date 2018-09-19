# This file is available under the terms of the MIT License

@doc """
Types for describing DMD fit and storing necessary 
variables
"""

"""
DMDParams - stores the description of the problem
including data and sizes. We 
only provide constructors for BLAS friendly types
for now
"""

mutable struct DMDParams{T<:AbstractFloat}
    # dimensions
    m::Integer
    n::Integer
    k::Integer
    # data
    X::Matrix{Complex{T}}
    x::Array{Vector{Complex{T}},1}
    t::Vector{Complex{T}}
end

# constructors
function DMDParams(k, X, t)
    m,n = size(X);
    x   = col_view(X);

    Tx  = eltype(X);
    Tt  = eltype(t);
    if Tx ≠ Tt
        t = convert(Vector{Tx}, t);
    end

    return DMDParams(m,n,k,X,x,t)
end


"""
DMDVars - stores both extra space needed for
the solution and intermediate computations and various
wrappers of these arrays and the original X data. We 
only provide constructors for BLAS friendly types
"""
mutable struct DMDVars{T<:AbstractFloat}
    # decision variables
    a::Vector{Complex{T}}
    B::Matrix{Complex{T}}
    b::Array{Vector{Complex{T}},1}
    P::Matrix{Complex{T}}
    # residual
    R::Matrix{Complex{T}}
    r::Array{Vector{Complex{T}},1}
    # real correspondent of alpha and B
    ar::Vector{T}
    br::Array{Vector{T},1}
end

# constructors
function DMDVars(params)
    m = params.m;
    n = params.n;
    k = params.k;

    T = eltype(params.X);

    # initial everything at 0
    a = zeros(T, k);
    B = zeros(T, k, n);
    R = zeros(T, m, n);
    P = zeros(T, m, k);

    b = col_view(B);
    r = col_view(R);

    ar = col_view_real(a);
    br = col_view_real(B);

    return DMDVars(a,B,b,P,R,r,ar,br)
end

mutable struct DMDSVars{T<:AbstractFloat}
    # variables used for solve B, b
    PQ::Matrix{Complex{T}}
    PR::Matrix{Complex{T}}
    tP::Vector{Complex{T}}
end

# constructors
function DMDSVars(params)
    m = params.m;
    n = params.n;
    k = params.k;
    T = eltype(params.X);

    PQ = zeros(T, m, k);
    PR = zeros(T, k, k);
    tP = zeros(T, k);

    return DMDSVars(PQ, PR, tP)
end
