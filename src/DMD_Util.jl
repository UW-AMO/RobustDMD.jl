export simDMD, dmdexactestimate, besterrperm, besterrperm_wi

# This file is available under the terms of the MIT License

using LinearAlgebra
using Random
using Printf
using Optim
using Munkres

@doc """
Utility functions for evaluating exponential basis, 
generating synthetic examples, evaluating derivatives,
etc. Many depend on BLAS for speed   
"""

###########################################################
# return the column view (complex or real) of a matrix
function col_view(X)
    ndims(X) == 1 && (return X);
    m,n = size(X);
    T   = eltype(X);
    s   = sizeof(T);
    p   = pointer(X);
    x   = Array{Vector{T}, 1}(UndefInitializer(), n);
    for i = 1:n
        x[i] = unsafe_wrap(Array, p, m);
        p += m*s;
    end
    return x
end

function col_view_real(X)
    # assert X is complex
    T = typeof(real(X[1]));
    p = convert(Ptr{T}, pointer(X));
    s = sizeof(T);
    if ndims(X) == 1
        m = length(X);
        x = unsafe_wrap(Array, p, 2*m);
        return x
    else
        m,n = size(X);
        x   = Array{Vector{T},1}(UndefInitializer(), n);
        for i = 1:n
            x[i] = unsafe_wrap(Array, p, 2*m);
            p += 2*m*s;
        end
        return x
    end
end

###########################################################
# populate exponential matrix
function update_P!(params)
    T  = eltype(params.X);
    BLAS.gemm!('N', 'T', T(1.0), params.t, params.a, T(0.0), params.P);
    map!(exp, params.P, params.P);
end

function update_P_general!(P, t, a, f)
    T  = eltype(a);
    c0 = zero(T);
    c1 = one(T);
    BLAS.gemm!('N', 'T', c1, t, a, c0, P);
    map!(f, P, P);
end

function vc2vr(vc)
    T = eltype(real(vc[1]));
    k = length(vc);
    #
    pc = pointer(vc);
    pr = convert(Ptr{T}, pc);
    vr = unsafe_wrap(Array, pr, k<<1)
    #
    return vr
end

function vr2vc(vr)
    T = Complex{eltype(vr)};
    k = length(vr);
    #
    pr = pointer(vr);
    pc = convert(Ptr{T}, pr);
    vc = unsafe_wrap(Array, pc, k>>1)
    return vc
end

# simulate a simple synthetic example
#------------------------------------------------------------------------------
function simDMD(m, n, k, T; seed=123)
    Random.seed!(seed);
    #
    # time and space vector
    t = complex(collect(range(T(0.0), stop=T(1.0), length=m)));
    s = complex(collect(range(T(-pi), stop=T(pi), length=n)));
    #
    # data matrix
    c0 = zero(Complex{T});
    c1 = one(Complex{T});
    #
    at = im*randn(T,k);       # temporal modes
    bt = complex((rand(T,k))*n/10.0);   # spatial modes
    # phit = exp(t⋅alphatᵀ), psit = sin(s⋅betatᵀ)
    Pt = zeros(Complex{T},m,k); update_P_general!(Pt, t, at, exp);
    Bt = zeros(Complex{T},k,n); update_P_general!(Bt, bt, s, sin);
    # X = Pt⋅Bt
    X = zeros(Complex{T}, m, n);
    BLAS.gemm!('N', 'N', c1, Pt, Bt, c0, X);
    #
    return X, t, at, Bt
end

# ###########################################################
# # closed form solution of B for least squares problem

function dmdl2B!(B, alpha, m, n, k, X, t)
    T  = eltype(X);
    epsmin = 1e2*eps(typeof(real(X[1])));
    c0 = zero(T);
    c1 = one(T);
    phi = zeros(T,m,k);
    update_P_general!(phi, t, alpha, exp);


    # stabilized least squares solution

    F = svd(phi,full=false)

    s1 = maximum(F.S)
    k2 = sum(F.S .> s1*epsmin)

    Y = zeros(T,k2,n)
    U = view(F.U,:,1:k2)
    Vt = view(F.Vt,1:k2,:)
    BLAS.gemm!('C','N',c1,U,X,c0,Y)
    Y ./= F.S[1:k2]
    BLAS.gemm!('C','N',c1,Vt,Y,c0,B)

end

# ###########################################################
# # exact and trapezoidal dmd --- for generating initial guess
function dmdexactestimate(m,n,k,X,t;dmdtype="exact",
                          uselowrank=false,niter=0)
    # use the trapezoidal rule and exact DMD
    # to estimate eigenvalues
    # Assumes that the times are in order,
    # i.e. that t[i] < t[i+1]
    T  = eltype(X);

    if (dmdtype == "exact")
        dt = t[2]-t[1]

        if uselowrank
            x1op = transpose(LinearOperator(X[1:end-1,:]))
            F = psvdfact(x1op,rank=k,sketch = :randn,
                         sketch_randn_niter = niter);
        else
            x1 = transpose(X[1:end-1,:])
            F = svd(x1,full=false);
        end
        u = F.U[:,1:k]; s = Diagonal(Array{T}(F.S[1:k]));
        v = F.Vt[1:k,:]';

        atilde = u'*(transpose(X[2:end,:])*v)/s
        a = eigvals(atilde)
        a = log.(a)/dt
        
        B = zeros(T,k,n)
        dmdl2B!(B,a,m,n,k,X,t)
        
    elseif (dmdtype == "trap")

        if uselowrank
            x1op = LinearOperator(X[1:end-1,:])
            x2op = LinearOperator(X[2:end,:])
            midop = transpose(x1op)+transpose(x2op)
            F = psvdfact(midop,rank=k,sketch=:randn,
                         sketch_randn_niter=niter)
        else
            mid = transpose(X[1:end-1,:] + X[2:end,:])
            F = svd(mid, full=false)
        end        
        
        u = F.U[:,1:k]
        s = Diagonal(Array{T}(F.S[1:k]))
        dt = Diagonal(Array{T}(2.0./(t[2:end]-t[1:end-1])))
        v = dt*(F.Vt[1:k,:]')

        atilde = u'*(transpose(X[2:end,:])*v - transpose(X[1:end-1,:])*v)/s
        a = eigvals(atilde)
        B = zeros(T,k,n)
        
        dmdl2B!(B,a,m,n,k,X,t)

    else

        error("unknown value for dmdtype")

    end

    return a, B
end


###########################################################
# error measure
function besterrperm(v1,v2)
    n = length(v1)
    A = Array{typeof(abs(v1[1]))}(undef,n,n)
    for j = 1:n
        for i = 1:n
            A[i,j] = abs(v1[i]-v2[j])
        end
    end
    if any(isnan,A)
        println("besterrperm: NaN in A, abort")
        return Inf
    end
      
    p = munkres(A)
    err = 0.0
    for i = 1:n
        err = err + abs(v1[i]-v2[p[i]])
    end
    return err
end

function besterrperm_wi(v1,v2)
    n = length(v1)
    A = Array{typeof(abs(v1[1]))}(undef,n,n)
    for j = 1:n
        for i = 1:n
            A[i,j] = abs(v2[i]-v1[j])
        end
    end
    p = munkres(A)
    err = 0.0
    for i = 1:n
        err = err + abs(v1[p[i]]-v2[i])
    end
    return err, p
end


