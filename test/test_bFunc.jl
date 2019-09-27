# unit test the BFunc and bFunc

include("../src/DMD_Util.jl");
include("../src/DMD_Type.jl");
include("../src/DMD_Func.jl");

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
	T = typeof(real(r[1]));
	return T(0.5)*sum(abs2, r)
end
#
function lossGrad(g, r)
end
#
# Create object
#------------------------------------------------------------------------------
params = DMDParams(k, X, t, lossFunc, lossGrad);
#
# Initialize with true parameter
#------------------------------------------------------------------------------
copyto!(params.a, at);
copyto!(params.B, Bt);
#
# Evaluate BFunc
#------------------------------------------------------------------------------
update_P!(params);
B_obj = BFunc(params);
#
if abs(B_obj) < 1e-6
	println("BFunc: OK")
else
	println("BFunc: Wrong, obj: $B_obj")
end
#
# Evaluate bFunc
#------------------------------------------------------------------------------
Random.randn!(params.R);
id = rand(1:n);
#
b_obj = bFunc(params, id);
#
if abs(b_obj) < 1e-6
	println("bFunc: OK")
else
	println("bFunc: Wrong, obj: $b_obj")
end