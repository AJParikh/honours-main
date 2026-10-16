#=

Generate cloud initial conditions from either a supplied state and independent
Gaussian perturbations or a matched OEM state and full six-dimensional covariance.

Currently this is rather fixed in terms of distributions used and quantities randomized but in the future it will be made more modular and flexible with such quantities being defined in a separate configuration file and taken as function inputs for the functions within this module.

=#

module ics

# Select the OEM source for generate_initial_conditions(N; ...) here.
OEM_FILENAME = "25544_ISS"

# -------------------
# ----- Imports -----
# -------------------

using LinearAlgebra, DifferentialEquations, StaticArrays, BenchmarkTools
using SatelliteToolbox, SatelliteToolboxAtmosphericModels, SatelliteToolboxBase
using Dates, Random, Distributions, Statistics
using Plots

include("terminal_formatting.jl")
using .tf

function generate_initial_conditions(u0::SVector{6, Float64}, N::Int; pn::Bool=false, vn::Bool=true,
    seed::Int=42, position_variance::Float64=500.0, velocity_variance::Float64=500.0)

    pn_cov = SMatrix{3, 3}(
        position_variance, 0.0, 0.0,
        0.0, position_variance, 0.0,
        0.0, 0.0, position_variance
    )
    vn_cov = SMatrix{3, 3}(
        velocity_variance, 0.0, 0.0,
        0.0, velocity_variance, 0.0,
        0.0, 0.0, velocity_variance
    )
    pnd = MvNormal(zeros(3), pn_cov)
    vnd = MvNormal(zeros(3), vn_cov)


    rng = MersenneTwister(seed)
    noisy_initial_conditions = repeat(reshape(collect(u0), 6, 1), 1, N)
    if pn
        noisy_initial_conditions[1:3, 2:N] .+= rand(rng, pnd, N - 1)
    end
    if vn
        noisy_initial_conditions[4:6, 2:N] .+= rand(rng, vnd, N - 1)
    end

    initial_mean = vec(mean(noisy_initial_conditions, dims=2))
    initial_covariance = cov(noisy_initial_conditions, dims=2)

    return noisy_initial_conditions, initial_mean, initial_covariance
end

# -------------------------------
# ----- OEM Initial Conditions -----
# -------------------------------

# Preserve all fractional-second digits when comparing epochs (no rounding).
function epoch_key(epoch::AbstractString)
    matched = match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?Z?$", epoch)
    isnothing(matched) && error("Unsupported OEM epoch: $epoch")
    seconds = BigInt(Dates.value(DateTime(matched[1]))) // BigInt(1000)
    fraction = matched[2]
    return isnothing(fraction) ? seconds : seconds + parse(BigInt, fraction) // big(10)^length(fraction)
end

parse_number(value) = parse(Float64, replace(value, 'D' => 'E', 'd' => 'e'))

"""
    read_initial_conditions(filename=OEM_FILENAME)

Read a CCSDS KVN OEM and select the earliest state/covariance pair with exactly
equal epochs within the same segment. Return epoch, state, full covariance,
reference frame, time system, and source path. Convert km and km/s to SI,
including all covariance blocks (multiply by 10^6).
Covariances in a different frame are rejected rather than silently mixed.
"""
function read_initial_conditions(filename::AbstractString=OEM_FILENAME)
    path = isabspath(filename) ? filename : joinpath(@__DIR__, "sources", filename)
    states = Dict{Rational{BigInt}, SVector{6,Float64}}()
    metadata = Dict{String,String}()
    in_metadata = false
    in_covariance = false
    covariance_epoch = ""
    covariance_frame = ""
    covariance = zeros(6, 6)
    row = 0
    best_key = nothing
    best = nothing

    for (line_number, raw) in enumerate(eachline(path))
        line = strip(raw)
        (isempty(line) || startswith(line, "COMMENT")) && continue
        if line == "META_START"
            empty!(metadata)
            empty!(states)
            in_metadata = true
        elseif line == "META_STOP"
            in_metadata = false
        elseif line == "COVARIANCE_START"
            in_covariance = true
            covariance_epoch = ""
            row = 0
        elseif line == "COVARIANCE_STOP"
            row == 6 || error("Incomplete covariance at line $line_number in $path")
            in_covariance = false
        elseif occursin('=', line)
            key, value = strip.(split(line, '='; limit=2))
            if in_metadata
                metadata[key] = value
            elseif in_covariance && key == "EPOCH"
                (isempty(covariance_epoch) || row == 6) || error("Incomplete covariance before line $line_number")
                covariance_epoch = value
                covariance_frame = get(metadata, "REF_FRAME", "")
                covariance = zeros(6, 6)
                row = 0
            elseif in_covariance && key == "COV_REF_FRAME"
                row == 0 || error("COV_REF_FRAME must precede covariance rows")
                covariance_frame = value
            end
        elseif in_covariance
            isempty(covariance_epoch) && error("Covariance has no epoch at line $line_number")
            row += 1
            row <= 6 || error("Too many covariance rows at line $line_number")
            values = parse_number.(split(line))
            length(values) == row || error("Expected $row lower-triangle entries at line $line_number")
            for column in 1:row
                covariance[row, column] = covariance[column, row] = values[column] * 1e6
            end
            if row == 6
                key = epoch_key(covariance_epoch)
                if haskey(states, key) && (isnothing(best_key) || key < best_key)
                    frame = get(metadata, "REF_FRAME", "")
                    isempty(frame) && error("Missing REF_FRAME in OEM segment")
                    covariance_frame == frame || error("Covariance frame $covariance_frame differs from state frame $frame; rotation is required")
                    time_system = get(metadata, "TIME_SYSTEM", "")
                    isempty(time_system) && error("Missing TIME_SYSTEM in OEM segment")
                    all(isfinite, covariance) || error("Nonfinite OEM covariance")
                    best_key = key
                    best = (epoch=covariance_epoch, state=states[key], covariance=copy(covariance),
                        reference_frame=frame, time_system=time_system, source_path=path)
                end
            end
        elseif !in_metadata && occursin(r"^\d{4}-", line)
            fields = split(line)
            length(fields) in (7, 10) || error("Expected epoch and six state components (optionally three accelerations) at line $line_number")
            state = SVector{6,Float64}(parse_number.(fields[2:7]) .* 1000.0)
            all(isfinite, state) || error("Nonfinite OEM state")
            key = epoch_key(fields[1])
            haskey(states, key) && states[key] != state && error("Conflicting states at $(fields[1])")
            states[key] = state
        end
    end
    in_covariance && error("Unclosed OEM covariance block")
    isnothing(best) && error("No exactly matching state and covariance epochs found in $path")
    return best
end

"""
    generate_initial_conditions(N; filename=OEM_FILENAME, seed=42)

Sample the full joint position/velocity Gaussian, retaining cross-covariances.
As in ics.jl, particle 1 is the nominal state and the return values are the
6×N cloud, its sample mean, and its sample covariance. Use
read_initial_conditions to inspect the selected epoch and source covariance.
"""
function generate_initial_conditions(N::Int; filename::AbstractString=OEM_FILENAME, seed::Int=42)
    N >= 2 || throw(ArgumentError("At least two particles are required for sample covariance"))
    initial = read_initial_conditions(filename)
    # Scale to a correlation matrix before factoring to handle mixed SI units.
    diagonal = diag(initial.covariance)
    all(>=(0), diagonal) || error("OEM covariance has negative variances")
    scale = [value > 0 ? sqrt(value) : 1.0 for value in diagonal]
    scaled = initial.covariance ./ (scale * scale')
    decomposition = eigen(Symmetric(scaled))
    tolerance = 100 * eps(Float64) * max(1.0, maximum(abs, decomposition.values))
    minimum(decomposition.values) >= -tolerance || error("OEM covariance is not positive semidefinite")
    factor = Diagonal(scale) * decomposition.vectors * Diagonal(sqrt.(max.(decomposition.values, 0.0)))
    cloud = repeat(reshape(collect(initial.state), 6, 1), 1, N)
    cloud[:, 2:N] .+= factor * randn(MersenneTwister(seed), 6, N - 1)
    return cloud, vec(mean(cloud, dims=2)), cov(cloud, dims=2)
end

end # module ics
