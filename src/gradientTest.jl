#====================================================================
    Gradient Test Fucntion
    translate Sasha's MATLAB code into Julia

    input:
        function handle fcn: ℝⁿ -> ℝ
        initial point x
        pert: maximum element-wise perturbation, e.g. 1e-3.

    output:
        err is the deviation, as explained in gradientTest.pdf,
        between 1 and the quantity that should be close to 1.

====================================================================#

function gradientTest(fcn, x0, pert)

    ϵ = pert*randn(size(x0));

    c = 0.1;
    scale!(ϵ, 1.0/c^2);
    println("Running gradient test:");

    f0, g0 = fcn(x0);
    x1 = zeros(x0);

    println("Change, result");
    for iter = 1:6
        scale!(ϵ, c);
        for i in eachindex(x0)
            x1[i] = x0[i] + ϵ[i];
        end
        f1, g1 = fcn(x1);

        err = 0.0;
        for i in eachindex(x0)
            err += (g0[i] + g1[i])*ϵ[i];
        end
        err = 1.0 - 0.5*err/(f1 - f0);
        @printf("‖ϵ‖: %1.5e, err %1.5e\n", vecnorm(ϵ), err);
    end
end
