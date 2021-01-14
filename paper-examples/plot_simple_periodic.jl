using JLD, Plots, Statistics

pyplot() # plotting backend

include("plot_defs.jl")

# plot the data for the simple periodic example

dirname = @__DIR__
fname = dirname * "/results/simple_periodic_pg_out.jld"
file = jldopen(fname,"r")

errs = read(file["errs"])
params = read(file["params"])

sigmas = params[1,:]

mederrs = dropdims(median(errs,dims=2),dims=2)

mederrs_l2 = mederrs[1,:]
mederrs_huber = mederrs[2,:]
mederrs_exact = mederrs[3,:]

ph = plot(sigmas, mederrs_exact, label="Exact DMD")
plot!(sigmas, mederrs_l2, label="Optimized DMD")
plot!(sigmas, mederrs_huber, label="Robust DMD (Huber)")
xaxis!("background noise",:log10)
yaxis!("error in recovered eigenvalues",:log10)
title!("Simple Periodic Example")

#  save figure
mkpath(dirname * "/figures")
fname = dirname * "/figures/simple_periodic_fig.pdf"
savefig(ph,fname)
