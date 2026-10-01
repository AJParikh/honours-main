using BenchmarkTools
using SatelliteToolbox, SatelliteToolboxAtmosphericModels, SatelliteToolboxBase
using Plots

SpaceIndices.init()
plotly()

# --- Fixed inputs (same as your example) ---
jd    = date_to_jd(2018, 6, 19, 18, 35, 0)
glat  = deg2rad(0)
glong = deg2rad(0)

# --- Altitude sweep: 100 → 1500 km at 1 km resolution ---
alts_m = 100_000.0 : 1_000.0 : 1_500_000.0
alts_km = alts_m ./ 1_000.0

# --- Density at each altitude ---import Pkg;
ρ = [AtmosphericModels.nrlmsise00(jd, h, glat, glong).total_density for h in alts_m]
ρ_exp = [AtmosphericModels.exponential(h) for h in alts_m]

# --- Plot ---
plt = plot(
    alts_km, ρ_exp,
    xlabel    = "Altitude [km]",
    ylabel    = "Total Density [kg/m³]",
    label     = "Exponential Atmosphere",
    title     = "NRLMSISE-00 Total Density vs Altitude\n glat=-22°, glong=-45°",
    yscale    = :log10,
    linewidth = 2,
    legend    = true,
    grid      = true,
    size      = (900, 550),
    color     = :red
)

plot!(plt,
    alts_km, ρ,
    linewidth = 2,
    color     = :blue,
    label     = "NRLMSISE-00",
)

gui(plt);

