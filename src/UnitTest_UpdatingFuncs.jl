# unit test the set up

include("DMDType.jl")
include("DMDUtil.jl")

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
# Test Updating P matrix
#--------------------------------------------------------------------
update_P!(vars, params);
@time update_P!(vars, params);
err = vecnorm(vars.P - exp.(t*vars.a.'));
tol = 1e-10;
if err < tol
	println("update_P! is OK.");
else
	println("update_P! is NOT OK!");
end

#--------------------------------------------------------------------
# Test Updating R matrix
#--------------------------------------------------------------------
update_R!(vars, params);
@time update_R!(vars, params);
err = vecnorm(vars.R - vars.P*vars.B + params.X);
tol = 1e-10;
if err < tol
	println("update_R! is OK.");
else
	println("update_R! is NOT OK!");
end

update_r!(vars, params, 1);
@time update_r!(vars, params, 1);
err = vecnorm(vars.r[1] - vars.P*vars.b[1] + params.x[1]);
tol = 1e-10;
if err < tol
	println("update_r! is OK.");
else
	println("update_r! is NOT OK!");
end

#--------------------------------------------------------------------
# Test Updating QR decomposition of P
#--------------------------------------------------------------------
update_PQR!(vars, params, svars);
@time update_PQR!(vars, params, svars);
err = vecnorm(svars.PQ*svars.PR - vars.P);
err = min(err, vecnorm(svars.PQ'*svars.PQ - eye(params.k)));
tol = 1e-10;
if err < tol
	println("update_PQR! is OK.");
else
	println("update_PQR! is NOT OK!");
end

#--------------------------------------------------------------------
# Test Updating B, b
#--------------------------------------------------------------------
update_B!(vars, params, svars); copy!(vars.B, B0);
@time update_B!(vars, params, svars); copy!(vars.B, B0);
err = vecnorm(vars.B - vars.P \ params.X);
tol = 1e-10;
if err < tol
	println("update_B! is OK.");
else
	println("update_B! is NOT OK!");
end

update_b!(vars, params, svars, 1); copy!(vars.B, B0);
@time update_b!(vars, params, svars, 1); copy!(vars.B, B0);
err = vecnorm(vars.b[1] - vars.P \ params.x[1]);
tol = 1e-10;
if err < tol
	println("update_b! is OK.");
else
	println("update_b! is NOT OK!");
end