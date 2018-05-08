#==========================================================
    DMD Utility Functions
==========================================================#

###########################################################
# update Functions
function updatephimat!(phi, t, alpha)
    c0 = zero(Complex{Float64});
    c1 = one(Complex{Float64});
    BLAS.gemm!('N','T',c1,t,alpha,c0,phi);
    for I in eachindex(phi)
        phi[I] = exp(phi[I]);
    end
end

# update Functions
function updatephipsi!(phi, t, alpha, f)
    c0 = zero(Complex{Float64});
    c1 = one(Complex{Float64});
    BLAS.gemm!('N','T',c1,t,alpha,c0,phi);
    for I in eachindex(phi)
        phi[I] = f(phi[I]);
    end
end

function dmd_alphagrad1!(gr,vars,params)
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
    pc = convert(Ptr{Complex{Float64}}, pr);
    gc = unsafe_wrap(Array, pc, params.k);
    # compute complex gradient
    lossg!(R);
    c1 = one(Complex{Float64})
    c0 = zero(Complex{Float64})
    temp = zeros(B)
    scale!(t,R)
    BLAS.gemm!('T','N',c1,phi,R,c0,temp)
    temp = temp.*B
    BLAS.sum!(gc,temp)

    dmd_alphar_cg2rg!(gr,params.k)

end

function dmd_alphar_cg2rg!(gr,k)
    scale!(gr, -2.0);
    BLAS.scal!(k,-1.0,gr,2);
end

function updateResidual!(vars, params)
    c1 = one(Complex{Float64});
    copy!(vars.R, params.X);
    BLAS.gemm!('N','N',c1,vars.phi,vars.B,-c1,vars.R);
end

function updateResidual_sub!(vars, params, id)
    c1 = one(Complex{Float64});
    copy!(vars.r[id], params.x[id]);
    BLAS.gemm!('N','N',c1,vars.phi,vars.b[id],-c1,vars.r[id]);
end

###########################################################
# generate function
function genDMD(m, n, k, sigma, mu; seed=123, mode=1, p = 0.1)

    srand(seed);

    # time and space vector
    t = complex(linspace(0.0,1.0,m));
    s = complex(linspace(-pi, pi,n));

    # data matrix
    c0 = zero(Complex{Float64});
    c1 = one(Complex{Float64});
    alphat = im*randn(k);       # temporal modes
    betat = c1*6.0*randn(k);   # spatial modes
    # phit = exp(t⋅alphatᵀ), psit = sin(s⋅betatᵀ)
    phit = zeros(Complex{Float64},m,k); updatephimat!(phit, t, alphat);
    psit = zeros(Complex{Float64},n,k); updatephipsi!(psit, s, betat, sin);
    
    # xclean = phit⋅psitᵀ
    xclean  = zeros(Complex{Float64},m,n);
    BLAS.gemm!('N','T',c1,phit,psit,c0,xclean);

    xdat = copy(xclean)

    if (mode == 2)
        ncol = sum( rand(n) .< p )
        icols = sample(1:n,ncol,replace=false)
        ikeep = zeros(m,n)
        ikeep[:,icols] = 1.0
        noise = sigma*randn(m,n) + mu*randn(m,n).*ikeep
    else
        noise = sigma*randn(m,n) + mu*randn(m,n).*( rand(m,n) .< p )
    end

    xdat = xclean + noise

    return xdat, xclean, t, alphat, betat
end

###########################################################
# closed form solution of B for least square problem
function dmdl2B!(B, alpha, m, n, k, X, t; epsmin=1.0e-12)
    c0 = zero(Complex{Float64});
    c1 = one(Complex{Float64});
    phi = zeros(Complex{Float64},m,k);
    updatephimat!(phi, t, alpha);


    # stabilized least squares solution

    F = svdfact(phi,thin=true)

    s1 = maximum(F[:S])
    k2 = sum(F[:S] .> s1*epsmin)

    Y = zeros(Complex{Float64},k2,n)
    U = view(F[:U],:,1:k2)
    Vt = view(F[:Vt],1:k2,:)
    BLAS.gemm!('C','N',c1,U,X,c0,Y)
    scale!(1./F[:S][1:k2],Y)
    BLAS.gemm!('C','N',c1,Vt,Y,c0,B)

end

###########################################################
function dmdexactestimate(m,n,k,X,t;dmdtype="trap")
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
        B = zeros(Complex{Float64},k,n)
        dmdl2B!(B,alpha,m,n,k,X,t)
        
    else

        dx = (transpose(X[2:end,:]) - transpose(X[1:end-1,:]))
        
        for j = 1:m-1
            dt = t[j+1]-t[j]
            for i = 1:n
                dx[i,j] = dx[i,j]/dt
            end
        end
        
        xin = 0.5*(transpose(X[1:end-1,:]) + transpose(X[2:end,:]))
        u, s, v = svd(xin,thin = true)
        u1 = u[:,1:k]
        s1 = diagm(s[1:k])
        v1 = v[:,1:k]
        atilde = u1'*dx*v1/s1
        alpha = eigvals(atilde)
        B = zeros(Complex{Float64},k,n)
        dmdl2B!(B,alpha,m,n,k,X,t)

    end

    return alpha, B
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
