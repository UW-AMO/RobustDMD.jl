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

# ############################################################
# # update residuals
# function update_R!(vars, params)
#     T  = eltype(params.X);
#     c0 = zero(T);
#     c1 = one(T);

#     BLAS.gemm!('N', 'N', c1, vars.P, vars.B, c0, vars.R);
#     broadcast!(-, vars.R, vars.R, params.X);
# end

# function update_r!(vars, params, id)
#     T  = eltype(params.X);
#     c0 = zero(T);
#     c1 = one(T);

#     BLAS.gemm!('N', 'N', c1, vars.P, vars.b[id], c0, vars.r[id]);
#     broadcast!(-, vars.r[id], vars.r[id], params.x[id]);
# end

# ############################################################
# # update the QR factorization of P
# function update_PQR!(vars, params, svars)
#     P = vars.P;
#     T = eltype(P);
#     PQ = svars.PQ;
#     PR = svars.PR;
#     tP = svars.tP;

#     c1 = one(T);
#     c0 = zero(T);

#     # calculate QR decomposition
#     copy!(PQ, P);
#     LAPACK.geqrf!(PQ, tP);
#     LAPACK.orgqr!(PQ, tP);
#     BLAS.gemm!('C', 'N', c1, PQ, P, c0, PR);
# end

# # solve upper triangular linear system
# function upper_solve!(PR, b)
#     k = length(b);
#     # backsubtitution
#     b[k] = b[k]/PR[k,k];
#     for i = k-1:-1:1
#         # calculate the rhs
#         for j = i+1:k
#             b[i] -= PR[i,j]*b[j];
#         end
#         b[i] = b[i]/PR[i,i];
#     end
# end


# ############################################################
# # update B, b
# function update_B!(vars, params, svars)
#     T  = eltype(params.X);
#     c0 = zero(T);
#     c1 = one(T);
#     BLAS.gemm!('C', 'N', c1, svars.PQ, params.X, c0, vars.B);
#     for i = 1:params.n
#         upper_solve!(svars.PR, vars.b[i]);
#     end
# end

# function update_b!(vars, params, svars, id)
#     T  = eltype(params.X);
#     c0 = zero(T);
#     c1 = one(T);
#     BLAS.gemv!('C', c1, svars.PQ, params.x[id], c0, vars.b[id]);
#     upper_solve!(svars.PR, vars.b[id]);
# end


# function grad_ar!(gr, vars, params, svars)
#     #
#     # Helper routine: following the inner solve, 
#     # this routine computes the gradient w.r.t alpha
#     #
#     # This routine assumes that alphar, B, and R 
#     # are up-to-date
#     #
#     # NOTE: this function overwrites R to save space
#     #

#     t = params.t;
#     P = vars.P;
#     B = vars.B;
#     R = vars.R;
#     tM = svars.tM;

#     T = eltype(params.X);

#     # wrap complex array around galphar
#     pr = pointer(gr);
#     pc = convert(Ptr{T}, pr);
#     gc = unsafe_wrap(Array, pc, params.k);
#     # compute complex gradient
#     c0 = zero(T);
#     c1 = T(0.5);
#     conj!(R);
#     broadcast!(*, R, R, t);
#     BLAS.gemm!('T', 'N', c1, P, R, c0, tM);
#     broadcast!(*, tM, tM, B);
#     BLAS.sum!(gc, tM);

#     gc2gr!(gr, params.k);
# end

# function grad_ar!(gr, vars, params, svars, id)
#     #
#     # Helper routine: following the inner solve, 
#     # this routine computes the gradient w.r.t alpha
#     #
#     # This routine assumes that alphar, B, and R 
#     # are up-to-date
#     #
#     # NOTE: this function overwrites R to save space
#     #

#     t = params.t;
#     P = vars.P;
#     b = vars.b;
#     r = vars.r;

#     T = eltype(params.X);

#     # wrap complex array around galphar
#     pr = pointer(gr);
#     pc = convert(Ptr{T}, pr);
#     gc = unsafe_wrap(Array, pc, params.k);
#     # compute complex gradient
#     c0 = zero(T);
#     c1 = T(0.5);
#     conj!(r[id]);
#     broadcast!(*, r[id], r[id], t);
#     BLAS.gemv!('T', c1, P, r[id], c0, gc);
#     broadcast!(*, gc, gc, b[id]);

#     gc2gr!(gr, params.k);
# end


# function gc2gr!(gr, k)
#     T = eltype(gr);
#     scale!(gr, T(-2.0));
#     BLAS.scal!(k, T(-1.0), gr, 2);
# end

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
    bt = c1*T(6.0)*randn(T,k);   # spatial modes
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
function dmdexactestimate(m,n,k,X,t;dmdtype="trap")
    # use the trapezoidal rule and exact DMD
    # to estimate eigenvalues
    # Assumes that the times are in order,
    # i.e. that t[i] < t[i+1]
    T  = eltype(X);
    Tr = typeof(real(X[1]));

    if (dmdtype == "exact")
        x1 = transpose(X[1:end-1,:])
        x2 = transpose(X[2:end,:])

        dt = t[2]-t[1]
        
        F = svd(x1,full=false);
        u1 = F.U[:,1:k]
        s1 = diagm(0 => F.S[1:k])
        v1 = F.V[:,1:k]
        atilde = u1'*x2*v1/s1
        alpha = eigvals(atilde)
        alpha = log(alpha)/dt
        
        B = zeros(T,k,n)
        dmdl2B!(B,alpha,m,n,k,X,t)
        
    else

        dx = (transpose(X[2:end,:]) - transpose(X[1:end-1,:]))
        
        for j = 1:m-1
            dt = t[j+1]-t[j]
            for i = 1:n
                dx[i,j] = dx[i,j]/dt
            end
        end
        
        xin = Tr(0.5)*(transpose(X[1:end-1,:]) + transpose(X[2:end,:]))
        
        F = svd(xin, full=false)
        u1 = F.U[:,1:k]
        s1 = diagm(0 => F.S[1:k])
        v1 = F.V[:,1:k]
        atilde = u1'*dx*v1/s1
        alpha = eigvals(atilde)
        B = zeros(T,k,n)
        
        dmdl2B!(B,alpha,m,n,k,X,t)

    end

    return alpha, B
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


