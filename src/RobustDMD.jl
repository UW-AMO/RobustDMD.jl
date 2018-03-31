# module wrapper

__precompile__()

module RobustDMD

using Munkres

## DMD Base
### DMDFunc
export DMDObj, DMDObj_sub, alphafunc, alphagrad!, alpha_fg!
export bfunc_sub, bgrad_sub!, alphagrad_sub!
### DMDLoss
export l2_func, l2_grad!, huber_func, huber_grad!
export st_func, st_grad!
### DMDType
export DMDParams, DMDVariables, DMDLSQ_options, DMDLSQ_vars
export DMDnull_options, DMDnull_vars
export DMDVPSolverVariables, DMDVPSolverVariablesBFGS
export DMDVPSolverVariablesLSQ, DMDVPSolverVariablesNull
### DMDUtil
export updatephimat!, updatephipsi!, dmd_alphagrad1!
export dmd_alphar_cg2rg, updateResidual!, updateResidual_sub!
export genDMD, genDMD, dmdl2B!, dmdexactestimate
export besterrperm, besterrperm_wi

## Generic
### BFGS
export BFGS_options, BFGS_vars, My_BFGS, exact_BFGS!
### gradientTest
export gradientTest

## Outer Solvers
### DMDProxGrad
export PG_options, PG_solve_DMD!, descent_PG!, armijo_PG!
### DMDSVRG
export SVRG_options, SVRGnullprox!, SVRG_solve_DMD!
### DMDTrim
export Trim_options, Trimnullprox!, Trim_solve_DMD!

## Inner Solvers
### DMDVPSolvers
export projb_BFGS!, projb_BFGS_subset!, projb_BFGS_sub!
export projb_LSQ!, projb_LSQ_subset!, projb_LSQ_sub!
export DMD_LSQ_solve!
export projb_null!, projb_null_subset!, projb_null_sub!


include("DMDProxGrad.jl")
include("DMDFunc.jl")
include("BFGS.jl")
include("DMDLoss.jl")
include("DMDSVRG.jl")
include("DMDTrim.jl")
include("DMDType.jl")
include("DMDUtil.jl")
include("gradientTest.jl")
include("OptimizerStats.jl")
include("DMDVPSolvers.jl")

end
