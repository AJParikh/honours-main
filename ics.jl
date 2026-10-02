#=

This file is used to generate the initial conditions for the cloud using random sampling and a predefined probability distribution. The ICs an be position perturbed, velocity perturbed, or both.

Currently this is rather fixed in terms of distributions used and quantities randomized but in the future it will be made more modular and flexible with such quantities being defined in a separate configuration file and taken as function inputs for the functions within this module.

=#

module ics

# -------------------
# ----- Imports -----
# -------------------

using LinearAlgebra, DifferentialEquations, StaticArrays, BenchmarkTools
using SatelliteToolbox, SatelliteToolboxAtmosphericModels, SatelliteToolboxBase
using Random, Distributions, Statistics
using Plots

include("terminal_formatting.jl")
using .tf

function generate_initial_conditions(u0::SVector{6, Float64}, N::Int; pn::Bool=false, vn::Bool=true)

    pn_cov = SMatrix{3, 3}(
        500.0, 0.0  , 0.0  ,
        0.0  , 500.0, 0.0  ,
        0.0  , 0.0  , 500.0
    )
    vn_cov = SMatrix{3, 3}(
        500.0, 0.0  , 0.0  ,
        0.0  , 500.0, 0.0  ,
        0.0  , 0.0  , 500.0
    )
    pnd = MvNormal(zeros(3), pn_cov)
    vnd = MvNormal(zeros(3), vn_cov)


    rng = MersenneTwister(42)
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

end # module ics