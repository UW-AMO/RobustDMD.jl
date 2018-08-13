# This file is available under the terms of the MIT License

@doc """
Utility functions for evaluating exponential basis, 
generating synthetic examples, evaluating derivatives,
etc. Many depend on BLAS for speed   
"""
###########################################################
# populate exponential matrix

for (elty) in (Float32,Float64)
    @eval begin

        function updatephimat!(phi::Array{Complex{$elty},2}, t::Array{Complex{$elty},1}, alpha::Array{Complex{$elty},1})
            c0 = zero(Complex{$elty});
            c1 = one(Complex{$elty});
            BLAS.gemm!('N','T',c1,t,alpha,c0,phi);
            for I in eachindex(phi)
                phi[I] = exp(phi[I]);
            end
        end
    end
end


# update Functions
for (elty) in (Float32,Float64)
    @eval begin

        function updatephipsi!(phi::Array{Complex{$elty},2}, t::Array{Complex{$elty},1}, alpha::Array{Complex{$elty},1}, f)
            c0 = zero(Complex{$elty});
            c1 = one(Complex{$elty});
            BLAS.gemm!('N','T',c1,t,alpha,c0,phi);
            for I in eachindex(phi)
                phi[I] = f(phi[I]);
            end
        end
    end
end

############################################################
# gradients, residuals, etc

for (elty) in (Float32,Float64)
    @eval begin
        function dmd_alphagrad1!(gr::Array{$elty},vars::DMDVariables{$elty},params::DMDParams{$elty})
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

for (elty) in (Float32,Float64)
    @eval begin
        function updateResidual!(vars::DMDVariables{$elty}, params::DMDParams{$elty})
            c1 = one(Complex{$elty});
            copy!(vars.R, params.X);
            BLAS.gemm!('N','N',c1,vars.phi,vars.B,-c1,vars.R);
        end
    end
end


for (elty) in (Float32,Float64)
    @eval begin
        function updateResidual_sub!(vars::DMDVariables{$elty}, params::DMDParams{$elty}, id)
            c1 = one(Complex{$elty});
            copy!(vars.r[id], params.x[id]);
            BLAS.gemm!('N','N',c1,vars.phi,vars.b[id],-c1,vars.r[id]);
        end
    end
end

###########################################################
# generate a simple synthetic example

for (elty) in (Float32,Float64)
    @eval begin
        function genDMD(m, n, k, sigma::$elty, mu::$elty; seed=123, mode=1, p = 0.1)

            srand(seed);

            # time and space vector
            t = complex(collect(linspace($elty(0.0),$elty(1.0),m)));
            s = complex(collect(linspace($elty(-pi), $elty(pi),n)));

            # data matrix
            c0 = zero(Complex{$elty});
            c1 = one(Complex{$elty});
            alphat = im*randn($elty,k);       # temporal modes
            betat = c1*$elty(6.0)*randn($elty,k);   # spatial modes
            # phit = exp(t⋅alphatᵀ), psit = sin(s⋅betatᵀ)
            phit = zeros(Complex{$elty},m,k); updatephimat!(phit, t, alphat);
            psit = zeros(Complex{$elty},n,k); updatephipsi!(psit, s, betat, sin);
            
            # xclean = phit⋅psitᵀ
            xclean  = zeros(Complex{$elty},m,n);
            BLAS.gemm!('N','T',c1,phit,psit,c0,xclean);

            xdat = copy(xclean)

            if (mode == 2)
                ncol = sum( rand($elty,n) .< p )
                icols = sample(1:n,ncol,replace=false)
                ikeep = zeros($elty,m,n)
                ikeep[:,icols] = one($elty)
                noise = sigma*randn($elty,m,n) + mu*randn($elty,m,n).*ikeep
            else
                noise = sigma*randn($elty,m,n) + mu*randn($elty,m,n).*( rand($elty,m,n) .< p )
            end

            xdat = xclean + noise

            return xdat, xclean, t, alphat, betat
        end
    end
end
###########################################################
# closed form solution of B for least squares problem

for (elty) in (Float32,Float64)
    @eval begin
        function dmdl2B!(B::Array{Complex{$elty},2}, alpha::Array{Complex{$elty},1}, m, n, k, X::Array{Complex{$elty},2}, t::Array{Complex{$elty},1}; epsmin::$elty=$elty(1e2)*eps($elty))
            c0 = zero(Complex{$elty});
            c1 = one(Complex{$elty});
            phi = zeros(Complex{$elty},m,k);
            updatephimat!(phi, t, alpha);


            # stabilized least squares solution

            F = svdfact(phi,thin=true)

            s1 = maximum(F[:S])
            k2 = sum(F[:S] .> s1*epsmin)

            Y = zeros(Complex{$elty},k2,n)
            U = view(F[:U],:,1:k2)
            Vt = view(F[:Vt],1:k2,:)
            BLAS.gemm!('C','N',c1,U,X,c0,Y)
            scale!(1./F[:S][1:k2],Y)
            BLAS.gemm!('C','N',c1,Vt,Y,c0,B)

        end
    end
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
