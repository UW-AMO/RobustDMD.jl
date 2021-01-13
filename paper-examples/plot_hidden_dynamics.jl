using JLD, Plots, Statistics

pyplot() # plotting backend

# plot the data for the simple periodic example

dirname = @__DIR__
fname = dirname * "/results/hidden_dynamics_out.jld"
file = jldopen(fname,"r")

errs = read(file["errs"])
params = read(file["params"])

sigmas = params[1,:]

mederrs = dropdims(median(errs,dims=2),dims=2)

# sparse noise test

tname = "sparse_noise"
mederrs_l2 = mederrs[1,1,:]
mederrs_huber = mederrs[2,1,:]
mederrs_trim = mederrs[3,1,:]
mederrs_exact = mederrs[4,1,:]

ph = plot(sigmas, mederrs_l2, label="Global L2")
plot!(sigmas, mederrs_huber, label="Huber")
plot!(sigmas, mederrs_trim, label="Trimming")
plot!(sigmas, mederrs_exact, label="Exact DMD")
xaxis!("background noise",:log10)
yaxis!("error in recovered eigenvalues",:log10)
title!("Sparse Large Deviations")

#  save figure
mkpath(dirname * "/figures")
fname = dirname * "/figures/hidden_dynamics_" * tname *"_fig.pdf"
savefig(ph,fname)

# broken sensors test

tname = "broken_sensor"
mederrs_l2 = mederrs[1,2,:]
mederrs_huber = mederrs[2,2,:]
mederrs_trim = mederrs[3,2,:]
mederrs_exact = mederrs[4,2,:]

ph = plot(sigmas, mederrs_l2, label="Global L2")
plot!(sigmas, mederrs_huber, label="Huber")
plot!(sigmas, mederrs_trim, label="Trimming")
plot!(sigmas, mederrs_exact, label="Exact DMD")
xaxis!("background noise",:log10)
yaxis!("error in recovered eigenvalues",:log10)
title!("Broken Sensors")

#  save figure
mkpath(dirname * "/figures")
fname = dirname * "/figures/hidden_dynamics_" * tname *"_fig.pdf"
savefig(ph,fname)

# bump test

tname = "bump"
mederrs_l2 = mederrs[1,3,:]
mederrs_huber = mederrs[2,3,:]
mederrs_trim = mederrs[3,3,:]
mederrs_exact = mederrs[4,3,:]

ph = plot(sigmas, mederrs_l2, label="Global L2")
plot!(sigmas, mederrs_huber, label="Huber")
plot!(sigmas, mederrs_trim, label="Trimming")
plot!(sigmas, mederrs_exact, label="Exact DMD")
xaxis!("background noise",:log10)
yaxis!("error in recovered eigenvalues",:log10)
title!("Confounding Bump")

#  save figure
mkpath(dirname * "/figures")
fname = dirname * "/figures/hidden_dynamics_" * tname *"_fig.pdf"
savefig(ph,fname)
