# This file is available under the terms of the MIT License

@doc """
Types for describing DMD fit and storing necessary 
variables
"""

"""
DMDParams - stores the description of the problem
including data, sizes, and loss functions. We 
only provide constructors for BLAS friendly types
for now
"""

mutable struct DMDParams{T<:AbstractFloat}
    # dimensions
    m::Integer
    n::Integer
    k::Integer
    # data
    X::Array{Complex{T}}
    x::Array{Array{Complex{T},1}}
    t::Array{Complex{T}}
    # loss functions
    lossf::Function     # function
    lossg::Function     # gradient
end

# constructors

function DMDParams end

for (elty) in (Float32,Float64)

    @eval begin
        function DMDParams(k, X::Array{Complex{$elty}}, t::Array{Complex{$elty}}, lossf, lossg)
            if ndims(X) == 1
                m = length(X);
                n = 1;
            else
                m,n = size(X);
            end
            x   = Array{Array{Complex{$elty},1}}(n);
            p   = pointer(X);
            s   = sizeof(Complex{$elty});
            for j = 1:n
                x[j] = unsafe_wrap(Array, p, m);
                p += m*s;
            end
            return DMDParams(m, n, k, X, x, t, lossf, lossg)
        end
    end
end

"""
DMDVariables - stores both extra space needed for
the solution and intermediate computations and various
wrappers of these arrays and the original X data. We 
only provide constructors for BLAS friendly types
"""
mutable struct DMDVariables{T<:AbstractFloat}
    # decision variables
    alpha::Array{Complex{T}}
    B::Array{Complex{T}}
    b::Array{Array{Complex{T},1}}
    phi::Array{Complex{T}}
    # residual
    R::Array{Complex{T}}
    r::Array{Array{Complex{T},1}}
    # real correspondent of alpha and B
    alphar::Array{T}
    br::Array{Array{T,1}}
end

# constructors

function DMDVariables end

for (elty) in (Float32,Float64)

    @eval begin
        function DMDVariables(alpha::Array{Complex{$elty}}, B::Array{Complex{$elty}}, params::DMDParams)
            # complex 0 and 1
            c0 = zero(Complex{$elty});
            c1 = one(Complex{$elty});
            # dimensions
            m  = params.m;
            n  = params.n;
            k  = params.k;
            t  = params.t;
            X  = params.X;
            # check sizes
            if ndims(B) == 1
                kb = length(B);
                nb = 1;
            else
                kb,nb = size(B);
            end
            @assert (kb == k && nb == n) "B is not of the correct dimensions"
            @assert (length(alpha) == k) "alpha is not of the correct length"
            @assert (eltype(X) == Complex{$elty}) "alpha and B should be same type as params.X"
                
            # renew alpha and B
            alpha0 = copy(alpha);
            B0 = copy(B);
            # allocate alphar
            pc = pointer(alpha0);
            pr = convert(Ptr{$elty}, pc);
            alphar = unsafe_wrap(Array, pr, 2*k);
            # allocate b and br
            b  = Array{Array{Complex{$elty},1}}(n);
            br = Array{Array{$elty,1}}(n);
            pc = pointer(B0);
            pr = convert(Ptr{$elty}, pc);
            sc = sizeof(Complex{$elty});
            sr = sizeof($elty);
            for j = 1:n
                b[j]  = unsafe_wrap(Array, pc, k);
                br[j] = unsafe_wrap(Array, pr, 2*k);
                pc+= sc*k;
                pr+= sr*2*k;
            end
            # update phi
            phi  = zeros(Complex{$elty},m,k);
            updatephipsi!(phi, t, alpha0, exp);
            # update R = phi⋅B - X and r
            R  = copy(X);
            BLAS.gemm!('N','N',c1,phi,B0,-c1,R);
            r  = Array{Array{Complex{$elty},1}}(n);
            pc = pointer(R);
            for j = 1:n
                r[j] = unsafe_wrap(Array, pc, m);
                pc += m*sc;
            end

            return DMDVariables(alpha0, B0, b, phi, R, r, alphar, br)
        end
    end
end
