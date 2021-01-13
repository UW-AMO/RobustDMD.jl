# hidden dynamics system runner.
#
# This file actually runs the hidden dynamics system problem
# and writes the results to file
#

using RobustDMD, Printf, Random, JLD, StatsBase

# most definitions are in another file

dirname = @__DIR__
include(dirname * "/example_hidden_dynamics_defs.jl")

################################################################
################################################################
################################################################
# Actually run the tests...


# make data

xclean, eigs, t, x = hidden_dynamics_system_data(nt=128)

# solver parameters

function prox_mr(ar)
    # project for numerical stability
    k = length(ar) >> 1;
    for i = 1:k
        ar[i<<1-1] = min(2.0, ar[i<<1-1]);
    end
end

tol = 1e-7;
ptf = 10000000;
itm = 1000;
optsl2 = DMD_PG_Options(itm=itm, tol=tol,
                        ptf=ptf,prox = prox_mr);
optshub = DMD_PG_Options(itm=itm, tol=tol,
                         ptf=ptf,prox = prox_mr);
optstrim = DMD_PG_Options(itm=itm, tol=tol,
                          ptf=ptf,prox = prox_mr);

# test parameters

seed = 8675309
Random.seed!(seed)

p = 0.05 # add spikes at this percentage of data points
k = 4 # rank of fit

nj = 5 # number of different levels of background noise
ntest = 1

nkeep = Integer(floor(0.8*length(x))) # number of sensors to keep when trimming

errs = Array{Float64}(undef,4,ntest,3,nj)
eigs_comp = Array{Complex{Float64}}(undef,k,4,ntest,3,nj)
params = Array{Float64}(undef,7,nj)

@printf "==================================================\n\n"
@printf "Running the hidden dynamics example... \n\n"

for j = 1:nj
    @printf "starting noise level %d of %d\n" j nj
    sigma = 10.0^(-6+j) # noise floor
    mu = 1.0 # size of spikes
    A = 1.0 # height of bump
    w = 10.0 # width of bump
    kappa = 5*sigma
    params[1,j] = sigma
    params[2,j] = mu
    params[3,j] = p
    params[4,j] = kappa
    params[5,j] = A
    params[6,j] = w
    params[7,j] = nkeep
    
    Threads.@threads for i = 1:ntest
        
        xdat = make_corrupt_data_sparse_spikes(xclean,sigma,mu,p)
        
        (err1, err2, err3, err4, eig1, eig2, eig3, eig4) = (
            dofits(xdat,t,eigs,kappa,optsl2,optshub,optstrim,nkeep))
        
        @printf "test %d %d sparse spikes\n"  j i
        @printf "errs %e %e %e %e\n" err1 err2 err3 err4
        errs[1,i,1,j] = err1
        errs[2,i,1,j] = err2
        errs[3,i,1,j] = err3
        errs[4,i,1,j] = err4        
        
        eigs_comp[:,1,i,1,j] = eig1[:]
        eigs_comp[:,2,i,1,j] = eig2[:]
        eigs_comp[:,3,i,1,j] = eig3[:]
        eigs_comp[:,4,i,1,j] = eig4[:]        

        xdat = make_corrupt_data_broken_sensors(xclean,sigma,mu,p)
        
        (err1, err2, err3, err4, eig1, eig2, eig3, eig4) = (
            dofits(xdat,t,eigs,kappa,optsl2,optshub,optstrim,nkeep))
        
        @printf "test %d %d broken sensors\n"  j i
        @printf "errs %e %e %e %e\n" err1 err2 err3 err4
        
        errs[1,i,2,j] = err1
        errs[2,i,2,j] = err2
        errs[3,i,2,j] = err3
        errs[4,i,2,j] = err4        
        
        eigs_comp[:,1,i,2,j] = eig1[:]
        eigs_comp[:,2,i,2,j] = eig2[:]
        eigs_comp[:,3,i,2,j] = eig3[:]
        eigs_comp[:,4,i,2,j] = eig4[:]        

        xdat = make_corrupt_data_bump(xclean,x,t,sigma,A,w)
        
        (err1, err2, err3, err4, eig1, eig2, eig3, eig4) = (
            dofits(xdat,t,eigs,kappa,optsl2,optshub,optstrim,nkeep))
        
        @printf "test %d %d bump\n"  j i
        @printf "errs %e %e %e %e\n" err1 err2 err3 err4
        
        errs[1,i,3,j] = err1
        errs[2,i,3,j] = err2
        errs[3,i,3,j] = err3
        errs[4,i,3,j] = err4        
        
        eigs_comp[:,1,i,3,j] = eig1[:]
        eigs_comp[:,2,i,3,j] = eig2[:]
        eigs_comp[:,3,i,3,j] = eig3[:]
        eigs_comp[:,4,i,3,j] = eig4[:]        

    end
end

fname = dirname * "/results/hidden_dynamics_out.jld"

@printf "writing results to file %s\n" fname

# save the results to file

mkpath(dirname * "/results")
file = jldopen(fname,"w")
file["errs"] = errs
file["eigs_comp"] = eigs_comp
file["params"] = params
close(file)

@printf "\n ...done\n"

@printf "==================================================\n"


