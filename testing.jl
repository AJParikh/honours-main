#=

This file is for the sole purpose of testing different segments of the code and will likely be deleted later.

=#

# -------------------
# ----- Imports -----
# -------------------

using LinearAlgebra, DifferentialEquations, StaticArrays, BenchmarkTools
using SatelliteToolbox, SatelliteToolboxAtmosphericModels, SatelliteToolboxBase
using Random, Distributions, Statistics
using Plots

include("terminal_formatting.jl")
using .tf

include("accelerations.jl")
using .accelerations

include("integrators.jl")
using .integrators

include("ics.jl")
using .ics

include("post_utils.jl")
using .post_utils

tf.TerminalNewRun("(testing.jl)")

# -------------------
# ----- Testing -----
# -------------------

benchmark::Bool = false
covariance::Bool = !benchmark
earthview::Bool = true

tf.TerminalHeading1("Propagator Settings")

propagation = integrators.Propagator(10.0, 1.0e5)
u0 = SVector(7.0e6, 0.0, 0.0, 0.0, 7.5e3, 1.0e3)

println("Time step: ", propagation.dt, " s")
println("Total propagation time: ", propagation.tend, " s")


if benchmark
    println("\e[1;31m" * "-"^50)
    println("Two-Body Perturbed")
    println("-"^50 * "\e[0m")

    model = accelerations.PerturbedGravModel(3.986004418e14, 1.08262668e-3, -2.53265648e-6, -1.61962159e-6, 6378137.0)

    println("\e[33mYoshida 4th order:\e[0m")
    t, pos, vel = integrators.Yoshida(u0, propagation, model, Val(2));
    @btime integrators.Yoshida($u0, $propagation, $model, Val(2));
    println("Final position: ", pos[:,end])
    println("Final velocity: ", vel[:,end])

    println("-"^50)

    println("\e[33mYoshida 8th order:\e[0m")
    t, pos, vel = integrators.Yoshida(u0, propagation, model, Val(4));
    @btime integrators.Yoshida($u0, $propagation, $model, Val(4));
    println("Final position: ", pos[:,end])
    println("Final velocity: ", vel[:,end])

    println("-"^50)
end

if covariance
    println()
    tf.TerminalHeading1("Covariance Propagation")

    ensemble_size = 10000

    println("Ensemble size: ", ensemble_size)

    noisy_initial_conditions, initial_position_mean, initial_position_covariance = ics.generate_initial_conditions(u0, ensemble_size; pn=true, vn=false)

    println()
    tf.TerminalHeading2("Initial Conditions Statistics")
    println("Noisy initial position mean: ")
    display(initial_position_mean)
    println()
    println("Noisy initial position covariance: ")
    display(initial_position_covariance)

    model = accelerations.PerturbedGravModel(
        3.986004418e14,
        1.08262668e-3,
        -2.53265648e-6,
        -1.61962159e-6,
        6378137.0,
    )
    end_positions = Matrix{Float64}(undef, 3, ensemble_size)
    for i in 1:ensemble_size
        initial_condition = SVector{6, Float64}(noisy_initial_conditions[:, i])
        _, positions, _ = integrators.Yoshida(initial_condition, propagation, model, Val(2))
        end_positions[:, i] = positions[:, end]
    end

    println()
    tf.TerminalHeading2("Propagated Positions Statistics")

    end_position_mean = vec(mean(end_positions, dims=2))
    end_position_covariance = cov(end_positions, dims=2)
    println("Propagated position mean: ")
    display(end_position_mean)
    println()
    println("Propagated position covariance: ")
    display(end_position_covariance)
    println()

    _, u0_positions, u0_velocities = integrators.Yoshida(u0, propagation, model, Val(2))
    u0_end_position = u0_positions[:, end]
    u0_end_velocity = u0_velocities[:, end]
    println("u0 propagated end position: ")
    display(u0_end_position)
    println()
    println("u0 propagated end velocity: ")
    display(u0_end_velocity)

    R = post_utils.Cart_to_VNB(SVector{3, Float64}(u0_end_position), SVector{3, Float64}(u0_end_velocity))

    println("VNB covariance matrix: ")
    cov_VNB = R * end_position_covariance * R'
    display(cov_VNB)
    display(cov_VNB[1,3]/(sqrt(cov_VNB[1,1]) * sqrt(cov_VNB[3,3])))

    noisy_initial_conditions[1:3, :] .-= u0[1:3]
    end_positions .-= u0_end_position

    v_hat = u0_end_velocity / norm(u0_end_velocity)
    n_hat = cross(u0_end_position, u0_end_velocity) / norm(cross(u0_end_position, u0_end_velocity))
    b_hat = cross(v_hat, n_hat)

    axes_scaling = 5000.0

    plotlyjs()
    plt = scatter3d(
        noisy_initial_conditions[1, :],
        noisy_initial_conditions[2, :],
        noisy_initial_conditions[3, :],
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
        title="Initial and Propagated Positions",
        legend=true,
        grid=true,
        size=(800, 600),
        extra_plot_kwargs=Dict(:layout => Dict(:scene => Dict(:aspectmode => "data"))),
        color=:blue,
        label="Initial positions",
        markerstrokecolor=:black,
        markersize=3,
        aspect_ratio=1
    )
    scatter3d!(
        plt,
        end_positions[1, :],
        end_positions[2, :],
        end_positions[3, :],
        color=:red,
        label="End positions",
        markerstrokecolor=:black,
        markersize=3,
    )
    plot!(
        plt,
        [0.0, 5 * axes_scaling * v_hat[1]],
        [0.0, 5 * axes_scaling * v_hat[2]],
        [0.0, 5 * axes_scaling * v_hat[3]],
        color=:green,
        linewidth=4,
        label="Velocity direction",
    )
    plot!(
        plt,
        [0.0, axes_scaling * n_hat[1]],
        [0.0, axes_scaling * n_hat[2]],
        [0.0, axes_scaling * n_hat[3]],
        color=:yellow,
        linewidth=4,
        label="Normal direction",
    )
    plot!(
        plt,
        [0.0, axes_scaling * b_hat[1]],
        [0.0, axes_scaling * b_hat[2]],
        [0.0, axes_scaling * b_hat[3]],
        color=:orange,
        linewidth=4,
        label="Binormal direction",
    )
    # sphere_radius = 6_371_000.0
    # sphere_theta = range(0, 2π, length=60)
    # sphere_phi = range(0, π, length=30)
    # sphere_x = sphere_radius .* cos.(sphere_theta) .* sin.(sphere_phi)'
    # sphere_y = sphere_radius .* sin.(sphere_theta) .* sin.(sphere_phi)'
    # sphere_z = sphere_radius .* ones(length(sphere_theta)) .* cos.(sphere_phi)'
    # surface!(
    #     plt,
    #     sphere_x,
    #     sphere_y,
    #     sphere_z,
    #     color=:lightgray,
    #     alpha=0.5,
    #     linealpha=0.1,
    #     label="Earth",
    # )
    display(plt)
end