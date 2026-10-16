using LinearAlgebra, DifferentialEquations, StaticArrays, BenchmarkTools
using SatelliteToolbox, SatelliteToolboxAtmosphericModels, SatelliteToolboxBase
using Random, Distributions, Statistics
using Plots
using DelimitedFiles
using ProgressMeter
using TOML

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

include(joinpath(@__DIR__, "rho_io.jl"))
using .rho_io

function main()

    plots_save::Bool = true
    plot_ψ::Bool = false

    tf.TerminalNewRun("(cloud_snapshots.jl)")

    plots_dir = joinpath(@__DIR__, "outputs", "plots")
    data_dir = joinpath(@__DIR__, "outputs", "data")
    mkpath(plots_dir)
    mkpath(data_dir)
    run_index = rho_io.next_index(data_dir)

    tf.TerminalHeading1("Propagator Settings")

    model = accelerations.PerturbedGravModel(
        3.986004418e14,
        1.08262668e-3,
        -2.53265648e-6,
        -1.61962159e-6,
        6378137.0,
    )

    total_time = 10.0e6
    propagation_step = 0.5e3
    snapshot_step = 5.0e5
    Δt = 10.0
    position_noise = true
    velocity_noise = false
    random_seed = 42
    position_variance = 500.0 # m² per Cartesian axis
    velocity_variance = 500.0 # (m/s)² per Cartesian axis

    propagation_step % Δt == 0.0 || error("Propagation step must be an integer multiple of Δt.")

    total_time > propagation_step || error("Total time must be greater than propagation step.")

    total_time % propagation_step == 0.0 || error("Total time must be an integer multiple of propagation step.")

    if plots_save
        isfinite(snapshot_step) && snapshot_step > 0 || error("Snapshot step must be positive and finite.")
        snapshot_step % propagation_step == 0.0 || error("Snapshot step must be an integer multiple of propagation_step.")
    end
    snapshot_stride = plots_save ? Int(round(snapshot_step / propagation_step)) : 1

    M = Int(round(total_time / propagation_step))
    update_i = Int(round(M / 20))
    propagation = integrators.Propagator(Δt, propagation_step)
    u0 = SVector{6, Float64}(7.0e6, 0.0, 0.0, 0.0, 7.5e3, 1.0e3)

    a = post_utils.Cart_to_Kep(u0, model.μ)[1]
    T = 2π * sqrt(a*a*a / model.μ)
    orbits = Int(round(total_time / T))

    println("Δt: ", propagation.dt, " s")
    println("Step propagation time: ", propagation.tend, " s")
    println("Total propagation time: ", total_time, " s")
    println("Orbits to be simulated: ", orbits)
    println()

    tf.TerminalHeading1("Covariance Propagation")

    N = 10000
    println("Ensemble size: ", N)

    u0_N, u0_N_μ, u0_N_Σ = ics.generate_initial_conditions(u0, N;
        pn=position_noise, vn=velocity_noise, seed=random_seed,
        position_variance=position_variance, velocity_variance=velocity_variance)

    # Store TOML as CSV comments so the output carries its own simulation context.
    metadata = Dict(
        "format_version" => 1,
        "units" => "SI (m, s, kg); angles in radians",
        "gravity_model" => Dict(
            "type" => string(nameof(typeof(model))),
            "parameters" => Dict(string(name) => getfield(model, name) for name in fieldnames(typeof(model)))),
        "propagation" => Dict(
            "integrator" => "Yoshida", "base_step" => "DKD",
            "generation" => 2, "order" => 4, # Matches Val(2) in the propagation loop.
            "dt_s" => Δt, "snapshot_interval_s" => propagation_step,
            "total_time_s" => total_time, "nominal_period_s" => T, "snapshot_step_s" => snapshot_step),
        "initial_conditions" => Dict(
            "source" => "normal_ics",
            "state_xyz_vxyz" => collect(u0), "ensemble_size" => N,
            "distribution" => "independent isotropic Gaussian position and velocity noise",
            "position_noise" => position_noise, "velocity_noise" => velocity_noise,
            "position_variance_m2" => position_variance,
            "velocity_variance_m2_s2" => velocity_variance,
            "rng" => "MersenneTwister", "seed" => random_seed,
            "first_particle_is_reference" => true,
            "sample_mean" => collect(u0_N_μ),
            "sample_covariance_rows" => [collect(row) for row in eachrow(u0_N_Σ)]),
        "diagnostics" => Dict("psi_BR_saved" => plot_ψ,
            "rho_VB" => "position correlation V-B in VNB",
            "rho_RT" => "position correlation R-T in RTN"))

    end_states = copy(u0_N)
    end_M_μ = Matrix{Float64}(undef, 6, M + 1)
    end_M_μ[:, 1] = u0_N_μ
    end_M_Σ = Array{Float64, 3}(undef, 6, 6, M + 1)
    end_M_Σ[:, :, 1] = u0_N_Σ

    R_VNB = post_utils.Cart_to_VNB(u0)
    u0_N_Σ_VNB = R_VNB * u0_N_Σ[1:3, 1:3] * R_VNB'
    end_M_ρ_VB = Vector{Float64}(undef, M + 1)
    end_M_ρ_VB[1] = u0_N_Σ_VNB[1,3]/(sqrt(u0_N_Σ_VNB[1,1]) * sqrt(u0_N_Σ_VNB[3,3]))

    R_RTN = post_utils.Cart_to_RTN(u0)
    u0_N_Σ_RTN = R_RTN * u0_N_Σ[1:3, 1:3] * R_RTN'
    end_M_ρ_RT = Vector{Float64}(undef, M + 1)
    end_M_ρ_RT[1] = u0_N_Σ_RTN[1,2]/(sqrt(u0_N_Σ_RTN[1,1]) * sqrt(u0_N_Σ_RTN[2,2]))

    if plot_ψ
        end_M_ψ_BR = Vector{Float64}(undef, M + 1)
        end_M_ψ_BR[1] = post_utils.ψ_BR(SVector{3, Float64}(u0[1:3]...), SVector{3, Float64}(u0[4:6]...))
    end

    u0_N_plot = copy(u0_N)
    u0_N_plot[1:3, :] .-= u0_N_plot[1:3, 1]

    plotlyjs()

    println("Beginning propagation of ensemble...")
    println()

    prog = Progress(M, dt=2.0)
    for i in 1:M
        Threads.@threads for j in 1:N
            ic = SVector{6, Float64}(end_states[:, j])
            u = integrators.Yoshida_final(ic, propagation, model, Val(2))
            @inbounds begin
                end_states[1,j] = u[1]; end_states[2,j] = u[2]; end_states[3,j] = u[3]
                end_states[4,j] = u[4]; end_states[5,j] = u[5]; end_states[6,j] = u[6]
            end
        end
        end_M_μ[:, i + 1] = vec(mean(end_states, dims=2))
        end_M_Σ[:, :, i + 1] = cov(end_states, dims=2)

        ref_pos = SVector{3, Float64}(end_states[1:3, 1])
        ref_vel = SVector{3, Float64}(end_states[4:6, 1])

        # tf.TerminalHeading2("$i timestep position XYZ-covariance")
        # display(end_M_Σ[1:3, 1:3, i])
        # println()

        # tf.TerminalHeading2("$i timestep position VNB-covariance")

        R_VNB = post_utils.Cart_to_VNB(ref_pos, ref_vel)
        cov_VNB = R_VNB * end_M_Σ[1:3, 1:3, i + 1] * R_VNB'
        # display(cov_VNB)
        # println()

        ρ_VB = cov_VNB[1,3]/(sqrt(cov_VNB[1,1]) * sqrt(cov_VNB[3,3]))
        end_M_ρ_VB[i + 1] = ρ_VB
        # println("V-B Correlation coefficient: ", ρ_VB)
        # println()

        # tf.TerminalHeading2("$i timestep position RTN-covariance")
        
        R_RTN = post_utils.Cart_to_RTN(ref_pos, ref_vel)
        cov_RTN = R_RTN * end_M_Σ[1:3, 1:3, i + 1] * R_RTN'
        # display(cov_RTN)
        # println()

        ρ_RT = cov_RTN[1,2]/(sqrt(cov_RTN[1,1]) * sqrt(cov_RTN[2,2]))
        end_M_ρ_RT[i + 1] = ρ_RT
        # println("R-T Correlation coefficient: ", ρ_RT)
        # println()

        if plot_ψ
            # tf.TerminalHeading2("$i timestep ψ_BR")

            end_M_ψ_BR[i + 1] = post_utils.ψ_BR(ref_pos, ref_vel)
            # println("ψ_BR: ", end_M_ψ_BR[i + 1])
            # println()
        end

        if plots_save && i % snapshot_stride == 0
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
            savefig(plt, joinpath(plots_dir, "snapshot_$(run_index)_$(i*propagation_step).html"))
        end
        next!(prog)
    end
    println()

    tf.TerminalHeading1("Propagation complete.")

    time_axis = (0:M) .* propagation_step
    data = plot_ψ ? hcat(time_axis, end_M_ρ_VB, end_M_ρ_RT, end_M_ψ_BR) : hcat(time_axis, end_M_ρ_VB, end_M_ρ_RT)

    csv_path = rho_io.save_evolution(data_dir, metadata, data, plot_ψ; index=run_index)

    tf.TerminalHeading1("ρ evolution data saved")
    println(csv_path)
end

using Profile

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
