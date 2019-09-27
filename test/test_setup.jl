# unit test the set up

include("../src/DMD_Util.jl");
include("../src/DMD_Type.jl");

# Generate DMD Synthetic Data
#------------------------------------------------------------------------------
# dimensions
m = 100;    # temporal dimension
n = 100;    # spatial dimension
k = 3;      # number of modes
T = Float64;
# generate data
X, t, at, Bt = simDMD(m, n, k, T; seed=123456);
# specify the loss
function lossFunc(r)
	T = typeof(real(r[0]));
	return T(0.5)*sum(abs2, r)
end
#
function lossGrad(g, r)
	copyto!(g, r);
end
#
# Create object
#------------------------------------------------------------------------------
params = DMDParams(k, X, t, lossFunc, lossGrad);
#
println("create DMD object: OK")