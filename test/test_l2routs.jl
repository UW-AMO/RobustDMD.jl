# unit test the bGrad

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
	return sum(abs2, r)
end
#
function lossGrad(r)
	conj!(r)
end
#
# Create object
#------------------------------------------------------------------------------
params = DMDParams(k, X, t, lossFunc, lossGrad, inner_directl2=true);
#
# Initialize with true parameter
#------------------------------------------------------------------------------
copyto!(params.a, at);
copyto!(params.B, Bt);
#
# Evaluate bGrad
#------------------------------------------------------------------------------

id = rand(1:params.n);


Random.randn!(params.B);
#
println(params.b[id])

err = aBFunc(params);
println(params.b[id])

#
if err < 1e-6
	println("aBFunc: OK");
else
	println("aBFunc: Wrong, err: $err");
end
#
#id = rand(1:params.n);
#
Random.randn!(params.b[id]);

println(params.b[id])

#
err = abFunc(params, id);

println(params.b[id])

#
if err < 1e-6
	println("abFunc: OK");
else
	println("abFunc: Wrong, err: $err");
end
