# This file is available under the terms of the MIT License

@doc """
Utility functions for evaluating exponential basis, 
generating synthetic examples, evaluating derivatives,
etc. Many depend on BLAS for speed   
"""
###########################################################
# return the column view (complex or real) of a matrix
function col_view(X)
    dim(X) == 1 && (return X);
    m,n = size(X);
    T   = eltype(X);
    s   = sizeof(T);
    p   = pointer(X);
    x   = Array{Vector{T}, 1}(n);
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
    if dim(X) == 1
        m = length(X);
        x = unsafe_wrap(Array, p, 2*m);
        return x
    else
        m,n = size(X);
        x   = Array{Vector{T},1}(n);
        for i = 1:n
            x[i] = unsafe_wrap(Array, p, 2*m);
            p += 2*m*s;
        end
        return x
    end
end

###########################################################
# populate exponential matrix
function update_P!(vars, params)
    T  = eltype(params.X);
    c0 = zero(T);
    c1 = one(T);
    BLAS.gemm!('N', 'T', c1,params.t, vars.a,c0, vars.P);
    map!(exp, vars.P, vars.P);
end

function update_P_general!(P, t, a, f)
    T  = eltype(a);
    c0 = zero(T);
    c1 = one(T);
    BLAS.gemm!('N', 'T', c1, t, a, c0, P);
    map!(f, P, P);
end

############################################################
# update residuals
function update_R!(vars, params)
    T  = eltype(params.X);
    c0 = zero(T);
    c1 = one(T);

    BLAS.gemm!('N', 'N', c1, vars.P, vars.B, c0, vars.R);
    broadcast!(-, vars.R, vars.R, params.X);
end

function update_r!(vars, params, id)
    T  = eltype(params.X);
    c0 = zero(T);
    c1 = one(T);

    BLAS.gemm!('N', 'N', c1, vars.P, vars.b[id], c0, vars.r[id]);
    broadcast!(-, vars.r[id], vars.r[id], params.x[id]);
end

############################################################
# update the QR factorization of P
function update_PQR(vars, params, svars)
    P = vars.P;
    T = eltype(P);
    PQ = svars.PQ;
    PR = svars.PR;
    tP = svars.tP;

    c1 = one(T);
    c0 = zero(T);

    # calculate QR decomposition
    copy!(PQ, P);
    LAPACK.geqrf!(PQ, tP);
    LAPACK.orgqr!(PQ, tP);
    BLAS.gemm!('C', 'N', c1, PQ, P, c0, PR);
end

# solve upper triangular linear system
function upper_solve!(PR, b)
    k = length(b);
    # backsubtitution
    b[k] = b[k]/PR[k,k];
    for i = k-1:-1:1
        # calculate the rhs
        for j = i+1:k
            b[i] -= P[i,j]*b[j];
        end
        b[i] = b[i]/P[i,i];
    end
end


############################################################
# update B, b
function update_B!(vars, params, svars)
    T  = eltype(params.X);
    c0 = zero(T);
    c1 = one(T);
    BLAS.gemm!('C', 'N', c1, svars.PQ, params.X, c0, vars.B);
    for i = 1:params.n
        upper_solve!(svars.PR, vars.b[i]);
    end
end

function update_b!(vars, params, svars, id)
    T  = eltype(params.X);
    c0 = zero(T);
    c1 = one(T);
    BLAS.gemv!('C', c1, svars.PQ, params.x[id], c0, vars.b[id]);
    upper_solve!(svars.PR, vars.b[id]);
end

for (elty) in (Float32,Float64)
    @eval begin
        function dmd_alphagrad1!(gr::Array{$elty},vars::DMDVars{$elty},params::DMDParams{$elty})
            #
            # Helper routine: following the inner solve, 
            # this routine computes the gradient w.r.t alpha
            #
            # This routine assumes that alphar, B, and R 
            # are up-to-date
            #
            # NOTE: this function overwrites R to save space
            #

            phi = vars.phi
            t = params.t
            B = vars.B
            R = vars.R
            lossg! = params.lossg

            # wrap complex array around galphar
            pr = pointer(gr);
            pc = convert(Ptr{Complex{$elty}}, pr);
            gc = unsafe_wrap(Array, pc, params.k);
            # compute complex gradient
            lossg!(R);
            c1 = one(Complex{$elty})
            c0 = zero(Complex{$elty})
            temp = zeros(B)
            scale!(t,R)
            BLAS.gemm!('T','N',c1,phi,R,c0,temp)
            temp = temp.*B
            BLAS.sum!(gc,temp)

            cg2rg!(gr,params.k)
        end
    end
end

for (elty) in (Float32,Float64)
    @eval begin
        function cg2rg!(gr::Array{$elty},k::Integer)
            scale!(gr, $elty(-2.0));
            BLAS.scal!(k,$elty(-1.0),gr,2);
        end
    end
end

###########################################################
# generate a simple synthetic example
function genDMD(m, n, k, sigma, mu; seed=123, mode=1, p = 0.1)

    T = typeof(sigma);
    srand(seed);

    # time and space vector
    t = complex(collect(linspace(T(0.0),T(1.0),m)));
    s = complex(collect(linspace(T(-pi), T(pi),n)));

    # data matrix
    c0 = zero(Complex{T});
    c1 = one(Complex{T});
    alphat = im*randn(T,k);       # temporal modes
    betat = c1*T(6.0)*randn(T,k);   # spatial modes
    # phit = exp(t⋅alphatᵀ), psit = sin(s⋅betatᵀ)
    phit = zeros(Complex{T},m,k); update_P_general!(phit, t, alphat, exp);
    psit = zeros(Complex{T},n,k); update_P_general!(psit, s, betat, sin);
    
    # xclean = phit⋅psitᵀ
    xclean  = zeros(Complex{T},m,n);
    BLAS.gemm!('N','T',c1,phit,psit,c0,xclean);

    xdat = copy(xclean)

    if (mode == 2)
        ncol = sum( rand(T,n) .< p )
        icols = sample(1:n,ncol,replace=false)
        ikeep = zeros(T,m,n)
        ikeep[:,icols] = one(T)
        noise = sigma*randn(T,m,n) + mu*randn(T,m,n).*ikeep
    else
        noise = sigma*randn(T,m,n) + mu*randn(T,m,n).*( rand(T,m,n) .< p )
    end

    xdat = xclean + noise

    return xdat, xclean, t, alphat, betat
end

###########################################################
# closed form solution of B for least squares problem

function dmdl2B!(B, alpha, m, n, k, X, t; epsmin=1e2*eps)
    T  = eltype(X);
    c0 = zero(Complex{T});
    c1 = one(Complex{T});
    phi = zeros(Complex{T},m,k);
    updatephimat!(phi, t, alpha);


    # stabilized least squares solution

    F = svdfact(phi,thin=true)

    s1 = maximum(F[:S])
    k2 = sum(F[:S] .> s1*epsmin)

    Y = zeros(Complex{T},k2,n)
    U = view(F[:U],:,1:k2)
    Vt = view(F[:Vt],1:k2,:)
    BLAS.gemm!('C','N',c1,U,X,c0,Y)
    scale!(1./F[:S][1:k2],Y)
    BLAS.gemm!('C','N',c1,Vt,Y,c0,B)

end

###########################################################
# exact and trapezoidal dmd --- for generating initial guess

for (elty) in (Float32,Float64)
    @eval begin

        function dmdexactestimate(m,n,k,X::Array{Complex{$elty}},t::Array{Complex{$elty}};dmdtype="trap")
            # use the trapezoidal rule and exact DMD
            # to estimate eigenvalues
            # Assumes that the times are in order,
            # i.e. that t[i] < t[i+1]

            if (dmdtype == "exact")
                x1 = transpose(X[1:end-1,:])
                x2 = transpose(X[2:end,:])

                dt = t[2]-t[1]
                
                u, s, v = svd(x1,thin = true)
                u1 = u[:,1:k]
                s1 = diagm(s[1:k])
                v1 = v[:,1:k]
                atilde = u1'*x2*v1/s1
                alpha = eigvals(atilde)
                alpha = log(alpha)/dt
                
                B = zeros(Complex{$elty},k,n)
                dmdl2B!(B,alpha,m,n,k,X,t)
                
            else

                dx = (transpose(X[2:end,:]) - transpose(X[1:end-1,:]))
                
                for j = 1:m-1
                    dt = t[j+1]-t[j]
                    for i = 1:n
                        dx[i,j] = dx[i,j]/dt
                    end
                end
                
                xin = $elty(0.5)*(transpose(X[1:end-1,:]) + transpose(X[2:end,:]))
                
                u, s, v = svd(xin,thin = true)
                u1 = u[:,1:k]
                s1 = diagm(s[1:k])
                v1 = v[:,1:k]
                atilde = u1'*dx*v1/s1
                alpha = eigvals(atilde)
                B = zeros(Complex{$elty},k,n)
                
                dmdl2B!(B,alpha,m,n,k,X,t)

            end

            return alpha, B
        end
    end
end

###########################################################
# error measure
function besterrperm(v1,v2)
    n = length(v1)
    A = Array{typeof(abs(v1[1]))}(n,n)
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
    A = Array{typeof(abs(v1[1]))}(n,n)
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
