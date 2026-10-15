using LinearAlgebra, DifferentialEquations, StaticArrays, BenchmarkTools
using SatelliteToolbox, SatelliteToolboxAtmosphericModels, SatelliteToolboxBase
using Random, Distributions, Statistics
using Plots
using DelimitedFiles
using ProgressMeter

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

    plots_save::Bool = false
    plot_ψ::Bool = false

    tf.TerminalNewRun("(cloud_snapshots.jl)")

    plots_dir = "./outputs/plots"
    data_dir = "./outputs/data"
    mkpath(plots_dir)
    mkpath(data_dir)

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
    Δt = 10.0

    propagation_step % Δt == 0.0 || error("Propagation step must be an integer multiple of Δt.")

    total_time > propagation_step || error("Total time must be greater than propagation step.")

    total_time % propagation_step == 0.0 || error("Total time must be an integer multiple of propagation step.")

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

    u0_N, u0_N_μ, u0_N_Σ = ics.generate_initial_conditions(u0, N; pn=true, vn=false)

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
    end_M_ρ_RT[1] = u0_N_Σ_RTN[1,3]/(sqrt(u0_N_Σ_RTN[1,1]) * sqrt(u0_N_Σ_RTN[3,3]))

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

        if plots_save
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
        next!(prog)
    end
    println()

    tf.TerminalHeading1("Propagation complete.")

    gr()

    time_axis = (0:M) .* propagation_step
    k = 1.0/50.0 # pi axis scaling

    plt = plot(
        time_axis,
        end_M_ρ_RT,
        xlabel = "Time [s]",
        ylabel = "end_M_ρ",
        title  = "end_M_ρ vs Time",
        label  = "end_M_ρ_RT",
        size = (1200, 400),
        dpi = 600,
        linewidth = 2,
        marker = :circle,
        markersize = 3,
        left_margin   = 8Plots.mm,
        right_margin  = 8Plots.mm,
        top_margin    = 4Plots.mm,
        bottom_margin = 8Plots.mm,
        legend = :topleft,
    )
    plot!(
        plt,
        time_axis,
        end_M_ρ_VB,
        label = "end_M_ρ_VB",
        linewidth = 2,
        marker = :circle,
        markersize = 3,
    )
    vline!(
        plt,
        collect(0.0:10*T:total_time),
        color = :green,
        linestyle = :dash,
        linewidth = 1,
        alpha = 0.5,
        label = "",
    )
    if plot_ψ
        plt_twin = twinx(plt)
        plot!(
            plt_twin,
            time_axis,
            end_M_ψ_BR,
            ylabel = "ψ_BR",
            label = "ψ_BR",
            color = :green,
            linewidth = 2,
            marker = :diamond,
            markersize = 3,
            legend = :topright,
            ylimits = (-k*π, k*π)
        )
    end

    savefig(plt, joinpath(plots_dir, "end_M_rho_progression.png"))

    tf.TerminalHeading1("ρ evolution plot done")

    data = plot_ψ ? hcat(time_axis, end_M_ρ_VB, end_M_ρ_RT, end_M_ψ_BR) : hcat(time_axis, end_M_ρ_VB, end_M_ρ_RT)

    open(joinpath(data_dir, "rho_progression.csv"), "w") do io
        println(io, plot_ψ ? "time_s,rho_VB,rho_RT,psi_BR" : "time_s,rho_VB,rho_RT")
        writedlm(io, data, ',')
    end

    tf.TerminalHeading1("ρ evolution data saved")
end

using Profile

main()