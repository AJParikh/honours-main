module accelerations

using LinearAlgebra, DifferentialEquations, StaticArrays, BenchmarkTools
using SatelliteToolbox, SatelliteToolboxAtmosphericModels, SatelliteToolboxBase

SpaceIndices.init()

struct GravModel
    μ::Float64
end

struct PerturbedGravModel
    μ::Float64
    c2::Float64
    c3::Float64
    c4::Float64
end
PerturbedGravModel(μ, J2, J3, J4, Re) = PerturbedGravModel(μ, 1.5*J2*Re^2, 2.5*J3*Re^3, 0.625*J4*Re^4)

struct Ellipsoid
    a::Float64      # semi-major axis
    b::Float64      # semi-minor axis
    e2::Float64     # first eccentricity squared
    ep2::Float64    # second eccentricity squared
    omf::Float64    # 1 - f
end

struct PerturbedGravDragModel
    μ::Float64
    c2::Float64
    c3::Float64
    c4::Float64
    cdrag::Float64
    jd0::Float64
    exp::Bool
end
PerturbedGravDragModel(μ, J2, J3, J4, Re, Cd, A, m, jd0, exp) = PerturbedGravDragModel(μ, 1.5*J2*Re^2, 2.5*J3*Re^3, 0.625*J4*Re^4, 0.5*Cd*A/m, jd0, exp)

function Ellipsoid(a, f)
    b   = a*(1 - f)
    e2  = f*(2 - f)
    return Ellipsoid(a, b, e2, e2/(1 - e2), 1 - f)
end

const WGS84 = Ellipsoid(6378137.0, 1/298.257223563)

@inline function ecef_to_geodetic(x::Float64, y::Float64, z::Float64, E::Ellipsoid = WGS84)
    lon = atan(y, x)
    p   = sqrt(muladd(x, x, y*y))

    # Polar axis: avoid p = 0 division
    if p < 1e-12 * E.a
        lat = copysign(0.5*π, z)
        return lat, lon, abs(z) - E.b
    end

    β = atan(z, E.omf*p)
    lat = 0.0
    @inbounds for _ in 1:3
        sβ, cβ = sincos(β)
        lat = atan(muladd(E.ep2*E.b*sβ*sβ, sβ, z),
                   muladd(-E.e2*E.a*cβ*cβ, cβ, p))
        β   = atan(E.omf*sin(lat), cos(lat))     # tanβ = (1-f) tanφ, quadrant-safe
    end

    sφ, cφ = sincos(lat)
    N_term = sqrt(muladd(-E.e2*sφ, sφ, 1.0))
    h = muladd(p, cφ, muladd(z, sφ, -E.a*N_term))
    return SVector(lat, lon, h)
end

@inline function expoatmo(h::Float64)
    h *= 1e-3

    if h <= 100.0
        return 0.0

    elseif h < 110.0
        return 5.297e-7 * exp(-(h - 100.0) / 5.877)

    elseif h < 120.0
        return 9.661e-8 * exp(-(h - 110.0) / 7.263)

    elseif h < 130.0
        return 2.438e-8 * exp(-(h - 120.0) / 9.473)

    elseif h < 140.0
        return 8.484e-9 * exp(-(h - 130.0) / 12.636)

    elseif h < 150.0
        return 3.845e-9 * exp(-(h - 140.0) / 16.149)

    elseif h < 180.0
        return 2.070e-9 * exp(-(h - 150.0) / 22.523)

    elseif h < 200.0
        return 5.464e-10 * exp(-(h - 180.0) / 29.740)

    elseif h < 250.0
        return 2.789e-10 * exp(-(h - 200.0) / 37.105)

    elseif h < 300.0
        return 7.248e-11 * exp(-(h - 250.0) / 45.546)

    elseif h < 350.0
        return 2.418e-11 * exp(-(h - 300.0) / 53.628)

    elseif h < 400.0
        return 9.518e-12 * exp(-(h - 350.0) / 53.298)

    elseif h < 450.0
        return 3.725e-12 * exp(-(h - 400.0) / 58.515)

    elseif h < 500.0
        return 1.585e-12 * exp(-(h - 450.0) / 60.828)

    elseif h < 600.0
        return 6.967e-13 * exp(-(h - 500.0) / 63.822)

    elseif h < 700.0
        return 1.454e-13 * exp(-(h - 600.0) / 71.835)

    elseif h < 800.0
        return 3.614e-14 * exp(-(h - 700.0) / 88.667)

    elseif h < 900.0
        return 1.17e-14 * exp(-(h - 800.0) / 124.64)

    elseif h < 1000.0
        return 5.245e-15 * exp(-(h - 900.0) / 181.05)

    else
        return 3.019e-15 * exp(-(h - 1000.0) / 268.00)
    end
end

@inline function diff(state::SVector{6, Float64}, g::GravModel, t)
    x, y, z, vx, vy, vz = state
    z2 = z*z
    r2 = muladd(x, x, muladd(y, y, z2))
    ir = inv(sqrt(r2))
    q  = ir*ir
    ir3 = ir*q
    k   = g.μ * ir3
    return SVector(
        vx, vy, vz,
        -k*x, -k*y, -k*z
    )
end

@inline function diff(state::SVector{6, Float64}, g::PerturbedGravModel, t)
    x, y, z, vx, vy, vz = state
    z2 = z*z
    r2 = muladd(x, x, muladd(y, y, z2))
    ir = inv(sqrt(r2))
    q  = ir*ir
    q2 = q*q
    ir3 = ir*q
    s2 = z2*q

    J2xy = g.c2 * q  * (5.0*s2 - 1.0)
    J3xy = g.c3 * q2 * z * (7.0*s2 - 3.0)
    J4xy = 3.0*g.c4 * q2 * (1.0 + s2*(-14.0 + 21.0*s2))
    J2z  = g.c2 * q  * (5.0*s2 - 3.0)
    J4z  = g.c4 * q2 * (15.0 + s2*(-70.0 + 63.0*s2))
    J3z  = 0.2 * g.μ * g.c3 * (ir3*q) * (3.0 + s2*(-30.0 + 35.0*s2))

    k   = g.μ * ir3
    Kxy = J2xy + J3xy + J4xy - 1.0
    return SVector(
        vx, vy, vz, 
        k*x*Kxy,
        k*y*Kxy,
        k*z*(J2z + J4z - 1.0) + J3z
    )
end

@inline function diff(state::SVector{6, Float64}, g::PerturbedGravDragModel, t)
    x, y, z, vx, vy, vz = state
    vxrel = vx + y * 7.2921159e-5
    vyrel = vy - x * 7.2921159e-5
    z2 = z*z
    r2 = muladd(x, x, muladd(y, y, z2))
    v = sqrt(muladd(vxrel, vxrel, muladd(vyrel, vyrel, vz*vz)))
    ir = inv(sqrt(r2))
    q  = ir*ir
    q2 = q*q
    ir3 = ir*q
    s2 = z2*q

    J2xy = g.c2 * q  * (5.0*s2 - 1.0)
    J3xy = g.c3 * q2 * z * (7.0*s2 - 3.0)
    J4xy = 3.0*g.c4 * q2 * (1.0 + s2*(-14.0 + 21.0*s2))
    J2z  = g.c2 * q  * (5.0*s2 - 3.0)
    J4z  = g.c4 * q2 * (15.0 + s2*(-70.0 + 63.0*s2))
    J3z  = 0.2 * g.μ * g.c3 * (ir3*q) * (3.0 + s2*(-30.0 + 35.0*s2))

    k   = g.μ * ir3
    Kxy = J2xy + J3xy + J4xy - 1.0

    jd = g.jd0 + t/86400
    θ  = jd_to_gmst(jd)
    lat, lon_eci, h = ecef_to_geodetic(x, y, z)
    lon = mod(lon_eci - θ + π, 2π) - π
    ρ = g.exp ? expoatmo(h) : AtmosphericModels.nrlmsise00(jd, h, lat, lon).total_density

    return SVector(
        vx, vy, vz, 
        k*x*Kxy - g.cdrag*ρ*v*vxrel,
        k*y*Kxy - g.cdrag*ρ*v*vyrel,
        k*z*(J2z + J4z - 1.0) + J3z - g.cdrag*ρ*v*vz
    )
end

#=
c is 1/2 * Cd * A / m
=#
@inline function diff(state::SVector{6, Float64}, c::Float64, t)
    x, y, z, vx, vy, vz = state
    vxrel = vx + y * 7.2921159e-5
    vyrel = vy - x * 7.2921159e-5
    v = sqrt(muladd(vxrel, vxrel, muladd(vyrel, vyrel, vz*vz)))

    jd = g.jd0 + t/86400
    θ  = jd_to_gmst(jd)
    lat, lon_eci, h = ecef_to_geodetic(x, y, z)
    lon = mod(lon_eci - θ + π, 2π) - π
    ρ = g.exp ? expoatmo(h) : AtmosphericModels.nrlmsise00(jd, h, lat, lon).total_density

    return SVector(
        vx, vy, vz, 
        -c*ρ*v*vxrel,
        -c*ρ*v*vyrel,
        -c*ρ*v*vz
    )
end

end # module


# const EarthGrav = GravModel(3.986004418e14)
# const EarthGravPerturbed = PerturbedGravModel(
#     3.986004418e14, 
#     1.08262668e-3, 
#     -2.53265648e-6, 
#     -1.61962159e-6,
#     6378137.0
# )
# const EarthGravPerturbedDrag = PerturbedGravDragModel(
#     3.986004418e14, 
#     1.08262668e-3, 
#     -2.53265648e-6, 
#     -1.61962159e-6,
#     6378137.0,
#     2.2,
#     1.0,
#     0.01,
#     date_to_jd(2018, 6, 19, 18, 35, 0),
#     true
# )

# @btime diff(SVector(7000e3, 0.0, 0.0, 0.0, 7.5e3, 1.0), $EarthGrav, 0.0);
# @btime diff(SVector(7000e3, 0.0, 0.0, 0.0, 7.5e3, 1.0), $EarthGravPerturbed, 0.0);
# @btime diff(SVector(7000e3, 0.0, 0.0, 0.0, 7.5e3, 1.0), $EarthGravPerturbedDrag, 0.0);