# unit test of objective and gradient

# unit test the set up

include("DMDType.jl")
include("DMDUtil.jl")
include("DMDFunc.jl")
include("gradientTest.jl")

#--------------------------------------------------------------------
# Generate DMD Synthetic Data
#--------------------------------------------------------------------
# dimensions
m = 100;    # temporal dimension
n = 100;    # spatial dimension
k = 3;      # number of modes
sigma = 1e-4;
mu = 100;
# generate data
X, Xt, t, alphat, betat = genDMD(m, n, k, sigma, mu; seed=123456);

#--------------------------------------------------------------------
# Initial Guess
#--------------------------------------------------------------------
# create initial guess using trap. rule with exact dmd
alpha0, B0 = dmdexactestimate(m,n,k,X,t);

#--------------------------------------------------------------------
# Create object
#--------------------------------------------------------------------
params = DMDParams(k, X, t);
vars = DMDVars(params);
svars = DMDSVars(params);
copy!(vars.a, alpha0);
copy!(vars.B, B0);

#--------------------------------------------------------------------
# Define function that return the objective and gradient
#--------------------------------------------------------------------
function my_fnc1(ar, vars, params, svars)
	gar = zeros(ar);
	obj = objective(ar, vars, params, svars);
	gradient!(ar, gar, vars, params, svars,
		updateP=false, updateB=false, updateR=false);
	return obj, gar
end
my_fnc1(ar) = my_fnc1(ar, vars, params, svars);

# apply the gradient test
println("Test objective and gradient for full function");
gradientTest(my_fnc1, randn(2*k), 1e-2);

#--------------------------------------------------------------------
# Define function that return the objective and gradient
#--------------------------------------------------------------------
function my_fnc2(ar, vars, params, svars, id)
	gar = zeros(ar);
	obj = objective(ar, vars, params, svars, id);
	gradient!(ar, gar, vars, params, svars, id,
		updateP=false, updateB=false, updateR=false);
	return obj, gar
end
id = 12;
my_fnc2(ar) = my_fnc2(ar, vars, params, svars, id);

# apply the gradient test
println("Test objective and gradient for partial function");
gradientTest(my_fnc2, randn(2*k), 1e-2);