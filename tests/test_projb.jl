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
params = DMDParams(k, X, t, lossFunc, lossGrad);
#
# Initialize with true parameter
#------------------------------------------------------------------------------
copyto!(params.a, at);
copyto!(params.B, Bt);
#
# Evaluate bGrad
#------------------------------------------------------------------------------
update_P!(params);
id = rand(1:n);
#
bt = copy(params.br[id]);
Random.randn!(params.br[id]);
#
projb(params, id);
#
err = sum(abs2, bt - params.br[id]);
#
if err < 1e-6
	println("projb: OK");
else
	println("projb: Wrong, err: $err");
end