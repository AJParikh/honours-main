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

function main()

    tf.TerminalNewRun("(cloud-snapshots.jl)")

    plots_dir = "./plots"
    mkpath(plots_dir)

    tf.TerminalHeading1("Propagator Settings")

    model = accelerations.PerturbedGravModel(
        3.986004418e14,
        1.08262668e-3,
        -2.53265648e-6,
        -1.61962159e-6,
        6378137.0,
    )

    total_time = 2.5e4
    propagation_step = 0.25e3
    M = Int(total_time / propagation_step)
    propagation = integrators.Propagator(10.0, propagation_step)
    u0 = SVector(7.0e6, 0.0, 0.0, 0.0, 7.5e3, 1.0e3)

    println("Δt: ", propagation.dt, " s")
    println("Step propagation time: ", propagation.tend, " s")
    println()

    tf.TerminalHeading1("Covariance Propagation")

    N = 10000
    println("Ensemble size: ", N)

    u0_N, u0_N_μ, u0_N_Σ = ics.generate_initial_conditions(u0, N; pn=true, vn=false)

    end_states = copy(u0_N)
    end_M_μ = Matrix{Float64}(undef, 6, M)
    end_M_μ[:, 1] = u0_N_μ
    end_M_Σ = Array{Float64, 3}(undef, 6, 6, M)
    end_M_Σ[:, :, 1] = u0_N_Σ
    end_M_ρ = Vector{Float64}(undef, M)
    end_M_ρ[1] = u0_N_Σ[1,3]/(sqrt(u0_N_Σ[1,1]) * sqrt(u0_N_Σ[3,3]))

    u0_N_plot = copy(u0_N)
    u0_N_plot[1:3, :] .-= u0_N_plot[1:3, 1]

    plotlyjs()

    println("Beginning propagation of ensemble...")

    for i in 1:M
        for j in 1:N
            ic = SVector{6, Float64}(end_states[:, j])
            _, pos, vel = integrators.Yoshida(ic, propagation, model, Val(2))
            end_states[:, j] = vcat(pos[:, end], vel[:, end])
        end
        end_M_μ[:, i] = vec(mean(end_states, dims=2))
        end_M_Σ[:, :, i] = cov(end_states, dims=2)

        ref_pos = SVector{3, Float64}(end_states[1:3, 1])
        ref_vel = SVector{3, Float64}(end_states[4:6, 1])

        # tf.TerminalHeading2("$i timestep position XYZ-covariance")
        # display(end_M_Σ[1:3, 1:3, i])
        # println()

        tf.TerminalHeading2("$i timestep position VNB-covariance")
        R = post_utils.Cart_to_VNB(ref_pos, ref_vel)
        cov_VNB = R * end_M_Σ[1:3, 1:3, i] * R'
        display(cov_VNB)
        println()
        ρ = cov_VNB[1,3]/(sqrt(cov_VNB[1,1]) * sqrt(cov_VNB[3,3]))
        end_M_ρ[i] = ρ
        println("V-B Correlation coefficient: ", ρ)
        println()

        v_hat = ref_vel / norm(ref_vel)
        n_hat = cross(ref_pos, ref_vel) / norm(cross(ref_pos, ref_vel))
        b_hat = cross(v_hat, n_hat)

        max_dist = maximum(
            norm(SVector(end_states[1, j], end_states[2, j], end_states[3, j]) - ref_pos)
            for j in 2:N
        )

        v_axes_scaling = max_dist * (1.2 + (0.3 * (M-i)/(M-1)))
        nb_axes_scaling = max_dist * (0.2 + (1.3 * (M-i)/(M-1)))

        end_states_plot = copy(end_states)
        end_states_plot[1:3, :] .-= ref_pos

        plt = scatter3d(
            u0_N_plot[1, :],
            u0_N_plot[2, :],
            u0_N_plot[3, :],
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
            end_states_plot[1, :],
            end_states_plot[2, :],
            end_states_plot[3, :],
            color=:red,
            label="End positions",
            markerstrokecolor=:black,
            markersize=3,
        )
        plot!(
            plt,
            [0.0, v_axes_scaling * v_hat[1]],
            [0.0, v_axes_scaling * v_hat[2]],
            [0.0, v_axes_scaling * v_hat[3]],
            color=:green,
            linewidth=4,
            label="Velocity direction",
        )
        plot!(
            plt,
            [0.0, nb_axes_scaling * n_hat[1]],
            [0.0, nb_axes_scaling * n_hat[2]],
            [0.0, nb_axes_scaling * n_hat[3]],
            color=:yellow,
            linewidth=4,
            label="Normal direction",
        )
        plot!(
            plt,
            [0.0, nb_axes_scaling * b_hat[1]],
            [0.0, nb_axes_scaling * b_hat[2]],
            [0.0, nb_axes_scaling * b_hat[3]],
            color=:orange,
            linewidth=4,
            label="Binormal direction",
        )
        savefig(plt, joinpath(plots_dir, "cloud_snapshot_$(i*propagation_step).html"))
    end

    plt = plot(
        1:M,
        end_M_ρ,
        xlabel = "Step index",
        ylabel = "end_M_ρ",
        title  = "end_M_ρ vs step",
        label  = "end_M_ρ",
        linewidth = 2,
        marker = :circle,
        markersize = 3,
    )

    savefig(plt, joinpath(plots_dir, "end_M_rho_progression.png"))

    tf.TerminalHeading1("Propagation complete.")
end

main()