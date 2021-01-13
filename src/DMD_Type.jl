import LinearAlgebra: BlasInt

export DMDParams, QRStore

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
    #
    # data
    X::Matrix{Complex{T}}
    x::Array{Vector{Complex{T}},1}
    t::Vector{Complex{T}}
    #
    # loss function and gradient
    lossFunc::Function
    lossGrad::Function
    #
    # variables
    a::Vector{Complex{T}}
    P::Matrix{Complex{T}}
    B::Matrix{Complex{T}}
    b::Array{Vector{Complex{T}},1}
    R::Matrix{Complex{T}}
    r::Array{Vector{Complex{T}},1}
    #
    ar::Vector{T}
    br::Array{Vector{T},1}
    # l2 direct storage
    qra::Union{Nothing,Vector{Complex{T}}}
    qrfact::Union{Nothing,QRPivoted{Complex{T},Array{Complex{T},2}}}
    qrTemp::Union{Nothing,Matrix{Complex{T}}}
    qrtemp::Union{Nothing,Array{Vector{Complex{T}},1}}
    
    # temp variables
    tv::Vector{Complex{T}}
    tM::Matrix{Complex{T}}
    tcnorm::Vector{T}
    idsort::Vector{S} where S <: Integer
    #
    # inner solve options
    inner_solver::Any
    inner_opts::Any
    inner_directl2::Bool

    # trimming
    nkeep::Integer
    ikeep::Any
end

# constructors
function DMDParams(k, X, t, lossFunc, lossGrad;
                   inner_directl2::Bool=false,
                   inner_opts=Optim.Options(),
                   inner_solver=BFGS(),
                   nkeep=size(X,2))
    m,n = size(X);
    x   = col_view(X);
    #
    Tx  = eltype(X);
    Tt  = eltype(t);
    if Tx != Tt
        t = convert(Vector{Tx}, t);
    end
    #
    a = zeros(Tx, k);
    P = zeros(Tx, m, k);
    B = zeros(Tx, k, n);
    R = zeros(Tx, m, n);
    #
    b = col_view(B);
    r = col_view(R);
    #
    ar = col_view_real(a);
    br = col_view_real(B);
    #
    tv = zeros(Tx, k);
    tM = zeros(Tx, k, n);
    tMkeep = zeros(Tx, k, nkeep);
    tcnorm = zeros(real(Tx),n)
    idsort = Array{Int64}(undef,n)
    #

    qra = nothing
    qrfact = nothing
    qrTemp = nothing
    qrtemp = nothing

    if inner_directl2
        qra = copy(a); qra .= NaN;
        qrfact = QRPivoted(copy(P),Array{Tx}(undef,min(m,k)),
                           zeros(BlasInt,k))
        qrTemp = copy(R)
        qrtemp = col_view(qrTemp)
    end
    
    ikeep = collect(1:nkeep)
    
    return DMDParams(m, n, k, X, x, t, lossFunc, lossGrad,
                     a, P, B, b,
                     R, r, ar, br,
                     qra, qrfact, qrTemp, qrtemp,
                     tv, tM, tcnorm, idsort,
                     inner_solver,
                     inner_opts, inner_directl2,
                     nkeep, ikeep)
end


