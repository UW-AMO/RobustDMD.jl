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
function func(br, params, id)
	copyto!(params.br[id], br);
	return bFunc(params, id)
end
func(br) = func(br, params, id);
#
function grad(gbr, br, params, id)
	copyto!(params.br[id], br);
	bGrad(gbr, params, id);
end
grad(gbr, br) = grad(gbr, br, params, id);
#
br = randn(k<<1);
gbr = zeros(k<<1);
grad(gbr, br);
#
# finite difference method verification
ϵ = 1e-8;
br_fd = zeros(k<<1);
gbr_fd = zeros(k<<1);
#
for i in eachindex(gbr_fd)
	copyto!(br_fd, br);
	br_fd[i] += ϵ;
	gbr_fd[i] = (func(br_fd) - func(br))/ϵ;
end
#
err = sum(abs2, gbr_fd - gbr);
if err < 1e-6
	println("bGrad: OK");
else
	println("bGrad: Wrong, err: $err");
end