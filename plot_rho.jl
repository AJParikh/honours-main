using Plots
using DelimitedFiles
using TOML

# Select outputs/data/rho_evo_<index>.csv to plot.
CSV_INDEX = 1

include(joinpath(@__DIR__, "terminal_formatting.jl"))
using .tf

function read_rho_data(csv_path::AbstractString)
    metadata_lines = String[]
    data, header = open(csv_path, "r") do io
        while !eof(io) && peek(io, Char) == '#'
            push!(metadata_lines, replace(readline(io), r"^# ?" => ""))
        end
        readdlm(io, ',', Float64; header=true)
    end
    isempty(metadata_lines) && error("CSV has no simulation metadata. Run cloud_snapshots.jl once to generate the new format.")
    metadata = TOML.parse(join(metadata_lines, "\n"))
    return data, vec(String.(header)), metadata
end

function main(index::Int=CSV_INDEX;
    data_dir::AbstractString=joinpath(@__DIR__, "outputs", "data"),
    plots_dir::AbstractString=joinpath(@__DIR__, "outputs", "plots"))
    index >= 0 || throw(ArgumentError("CSV index must be nonnegative"))
    csv_path = joinpath(data_dir, "rho_evo_$(index).csv")
    data, columns, metadata = read_rho_data(csv_path)
    required = ["time_s", "rho_VB", "rho_RT"]
    all(name -> name in columns, required) || error("CSV must contain time_s, rho_VB, and rho_RT columns.")
    size(data, 1) > 0 || error("CSV contains no data rows.")

    time_axis = data[:, findfirst(==("time_s"), columns)]
    end_M_ρ_VB = data[:, findfirst(==("rho_VB"), columns)]
    end_M_ρ_RT = data[:, findfirst(==("rho_RT"), columns)]
    plot_ψ = "psi_BR" in columns
    if plot_ψ
        end_M_ψ_BR = data[:, findfirst(==("psi_BR"), columns)]
    end

    # Derived by the simulation; plotting needs no gravity or orbital calculations.
    T = metadata["propagation"]["nominal_period_s"]
    isfinite(T) && T > 0 || error("Metadata nominal_period_s must be positive and finite.")

    mkpath(plots_dir)
    gr()

    total_time = last(time_axis)
    k = 1.0/50.0 # pi axis scaling

    plt = plot(
        time_axis,
        end_M_ρ_RT,
        xlabel = "Time [s]",
        ylabel = "ρ",
        title  = "ρ vs Time",
        label  = "ρ_RT",
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
        label = "ρ_VB",
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

    savefig(plt, joinpath(plots_dir, "rho_evo_plot_$(index).png"))

    tf.TerminalHeading1("ρ evolution plot done")

    return plt
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(isempty(ARGS) ? CSV_INDEX : parse(Int, ARGS[1]))
end
