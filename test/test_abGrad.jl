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
# Initialize with random vector
#------------------------------------------------------------------------------
copyto!(params.a, at);
copyto!(params.B, Bt);
#
# Evaluate abGrad
#------------------------------------------------------------------------------
id = rand(1:n);
function func(ar, params, id)
	copyto!(params.ar, ar);
	return abFunc(params, id)
end
func(ar) = func(ar, params, id);
#
function grad(gar, ar, params, id)
	copyto!(params.ar, ar);
	abGrad(gar, params, id);
end
grad(gar, ar) = grad(gar, ar, params, id);
#
ar = randn(k<<1);
gar = zeros(k<<1);
grad(gar, ar);
#
# finite difference method verification
ϵ = 1e-8;
ar_fd = zeros(k<<1);
gar_fd = zeros(k<<1);
#
for i in eachindex(gar_fd)
	copyto!(ar_fd, ar);
	ar_fd[i] += ϵ;
	gar_fd[i] = (func(ar_fd) - func(ar))/ϵ;
end
#
err = sum(abs2, gar_fd - gar);
if err < 1e-6
	println("abGrad: OK");
else
	println("abGrad: Wrong, err: $err");
end
#
# Evaluate aBGrad
#------------------------------------------------------------------------------
function func(ar, params)
	copyto!(params.ar, ar);
	return aBFunc(params)
end
func(ar) = func(ar, params);
#
function grad(gar, ar, params)
	copyto!(params.ar, ar);
	aBGrad(gar, params);
end
grad(gar, ar) = grad(gar, ar, params);
#
ar = randn(k<<1);
gar = zeros(k<<1);
grad(gar, ar);
#
# finite difference method verification
ϵ = 1e-8;
ar_fd = zeros(k<<1);
gar_fd = zeros(k<<1);
#
for i in eachindex(gar_fd)
	copyto!(ar_fd, ar);
	ar_fd[i] += ϵ;
	gar_fd[i] = (func(ar_fd) - func(ar))/ϵ;
end
#
err = sum(abs2, gar_fd - gar);
if err < 1e-6
	println("aBGrad: OK");
else
	println("aBGrad: Wrong, err: $err");
end
#
# Evaluate aBGrad with trimming
#------------------------------------------------------------------------------


params = DMDParams(k, X, t, lossFunc, lossGrad, nkeep=80);

#
# Initialize with random vector
#------------------------------------------------------------------------------
copyto!(params.a, at);
copyto!(params.B, Bt);

trimstandard(params)

function func(ar, params)
	copyto!(params.ar, ar);
	return aBFunc(params)
end
func(ar) = func(ar, params);
#
function grad(gar, ar, params)
	copyto!(params.ar, ar);
	aBGrad(gar, params);
end
grad(gar, ar) = grad(gar, ar, params);
#
ar = randn(k<<1);
gar = zeros(k<<1);
grad(gar, ar);
#
# finite difference method verification
ϵ = 1e-8;
ar_fd = zeros(k<<1);
gar_fd = zeros(k<<1);
#
for i in eachindex(gar_fd)
	copyto!(ar_fd, ar);
	ar_fd[i] += ϵ;
	gar_fd[i] = (func(ar_fd) - func(ar))/ϵ;
end
#
err = sum(abs2, gar_fd - gar);
if err < 1e-6
	println("aBGrad trim: OK");
else
	println("aBGrad: Wrong, err: $err");
end
