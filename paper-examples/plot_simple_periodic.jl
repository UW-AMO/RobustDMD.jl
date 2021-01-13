using JLD, Plots, Statistics


file = jldopen("results/simple_periodic_pg_out.jld","r")
errs = read(file["errs"])
params = read(file["params"])

sigmas = params[1,:]

mederrs = dropdims(median(errs,dims=2),dims=2)

mederrs_l2 = mederrs[1,:]
mederrs_huber = mederrs[2,:]
mederrs_exact = mederrs[3,:]

plot(sigmas, mederrs_l2, label="Global L2")
plot!(sigmas, mederrs_huber, label="Huber")
plot!(sigmas, mederrs_exact, label="Exact DMD")
xaxis!("background noise",:log10)
yaxis!("error in recovered eigenvalues",:log10)
