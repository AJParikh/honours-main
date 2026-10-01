module integrators

# -------------------
# ----- Imports -----
# -------------------

using LinearAlgebra, DifferentialEquations, StaticArrays, BenchmarkTools
using SatelliteToolbox, SatelliteToolboxAtmosphericModels, SatelliteToolboxBase

using ..accelerations

# --------------------------
# ----- Initialisation -----
# --------------------------

SpaceIndices.init()

struct Propagator
    dt::Float64
    tend::Float64
end

# --------------------------
# ----- Pure Numerical -----
# --------------------------

function numerical(u0::SVector{6, Float64}, prop_model::Propagator, grav_model, solver)
    prob = ODEProblem(accelerations.diff, u0, (0.0, prop_model.tend), grav_model)
    sol = solve(prob, solver; reltol=1e-14, abstol=1e-14, saveat=prop_model.dt, dense=false, save_everystep=false, calck=false)
    arr = Array(sol)
    pos = @view arr[1:3, :]
    vel = @view arr[4:6, :]

    return sol.t, pos, vel
end

# -----------------------
# ----- Forest-Ruth -----
# -----------------------

const FOREST_RUTH_COEFFS = let
    α = 1.0 / (2.0 * (2.0 - 2.0^(1/3)))
    β = (1.0 - 2.0^(1/3)) / (2.0 * (2.0 - 2.0^(1/3)))
    c = (α, β, β, α)                    # use tuples – immutable and fast
    d = (2α, -2^(1/3)*2α, 2α, 0.0)
    (c, d)
end

@inline function ForestRuthStep(u::SVector{6, Float64}, dt::Float64, grav_model)
    c, d = FOREST_RUTH_COEFFS
    x, y, z, vx ,vy, vz = u
    @inbounds for i in 1:4
        x+= c[i] * vx * dt
        y+= c[i] * vy * dt
        z+= c[i] * vz * dt
        du = accelerations.diff(SVector(x, y, z, vx, vy, vz), grav_model, 0.0)
        vx+= d[i] * du[4] * dt
        vy+= d[i] * du[5] * dt
        vz+= d[i] * du[6] * dt
    end
    return SVector{6,Float64}(x, y, z, vx, vy, vz)
end

function ForestRuth(u0::SVector{6, Float64}, prop_model::Propagator, grav_model)
    dt = prop_model.dt
    u = u0
    nsteps = Int(round(prop_model.tend / dt))
    pos = Matrix{Float64}(undef, 3, nsteps + 1)
    vel = Matrix{Float64}(undef, 3, nsteps + 1)
    @inbounds begin
        pos[1,1] = u0[1]
        pos[2,1] = u0[2]
        pos[3,1] = u0[3]
        vel[1,1] = u0[4]
        vel[2,1] = u0[5]
        vel[3,1] = u0[6]
    end

    @inbounds for i in 1:nsteps
        u = ForestRuthStep(u, dt, grav_model)
        pos[1,i+1] = u[1]
        pos[2,i+1] = u[2]
        pos[3,i+1] = u[3]
        vel[1,i+1] = u[4]
        vel[2,i+1] = u[5]
        vel[3,i+1] = u[6]
    end

    t = range(0.0, step=dt, length=nsteps+1)
    return t, pos, vel
end

# --------------------------------------------
# ----- Symplectic Yoshida (2n-th Order) -----
# --------------------------------------------

@inline function kdk_step(u::SVector{6,Float64}, grav_model, h::Float64)
    du1 = accelerations.diff(u, grav_model, 0.0)
    a1  = SVector{3,Float64}(du1[4], du1[5], du1[6])
    v_half = SVector{3,Float64}(u[4],u[5],u[6]) + 0.5h * a1
    r_new  = SVector{3,Float64}(u[1],u[2],u[3]) + h * v_half

    u_mid  = SVector{6,Float64}(r_new[1], r_new[2], r_new[3], v_half[1], v_half[2], v_half[3])
    du2 = accelerations.diff(u_mid, grav_model, 0.0)
    a2  = SVector{3,Float64}(du2[4], du2[5], du2[6])
    v_new = v_half + 0.5h * a2

    return SVector{6,Float64}(r_new[1], r_new[2], r_new[3], v_new[1], v_new[2], v_new[3])
end

@inline function dkd_step(u::SVector{6,Float64}, grav_model, h::Float64)
    r = SVector{3,Float64}(u[1],u[2],u[3])
    v = SVector{3,Float64}(u[4],u[5],u[6])

    r_half = r + 0.5h * v
    u_half = SVector{6,Float64}(r_half[1], r_half[2], r_half[3], v[1], v[2], v[3])
    du = accelerations.diff(u_half, grav_model, 0.0)
    a  = SVector{3,Float64}(du[4], du[5], du[6])
    v_new = v + h * a
    r_new = r_half + 0.5h * v_new

    return SVector{6,Float64}(r_new[1], r_new[2], r_new[3], v_new[1], v_new[2], v_new[3])
end

recursive_yoshida_step(u::SVector{6,Float64}, grav_model, h::Float64, ::Val{1}) = dkd_step(u, grav_model, h)

@inline function recursive_yoshida_step(u::SVector{6,Float64}, grav_model, h::Float64, ::Val{N}) where {N}
    k = 1.0 / (2N - 1)
    g1 =  1.0 / (2.0 - 2.0^k)
    g2 = 1.0 - 2.0 * g1

    u1 = recursive_yoshida_step(u,  grav_model, g1 * h, Val(N-1))
    u2 = recursive_yoshida_step(u1, grav_model, g2 * h, Val(N-1))
    u3 = recursive_yoshida_step(u2, grav_model, g1 * h, Val(N-1))
    return u3
end

function Yoshida(u0::SVector{6,Float64}, prop_model::Propagator, grav_model, ::Val{N}=Val(1)) where {N}
    dt = prop_model.dt
    n  = Int(round(prop_model.tend / dt))

    pos = Matrix{Float64}(undef, 3, n + 1)
    vel = Matrix{Float64}(undef, 3, n + 1)

    u = u0
    @inbounds begin
        pos[1,1], pos[2,1], pos[3,1] = u[1], u[2], u[3]
        vel[1,1], vel[2,1], vel[3,1] = u[4], u[5], u[6]
        for i in 1:n
            u = recursive_yoshida_step(u, grav_model, dt, Val(N))
            pos[1,i+1], pos[2,i+1], pos[3,i+1] = u[1], u[2], u[3]
            vel[1,i+1], vel[2,i+1], vel[3,i+1] = u[4], u[5], u[6]
        end
    end

    t = range(0.0, step=dt, length=n+1)
    return t, pos, vel
end

# ----------------------------------------
# ----- Strang Splitting (2nd Order) -----
# ----------------------------------------



end # module integrators