module post_utils

using LinearAlgebra, DifferentialEquations, StaticArrays, BenchmarkTools

@inline function Cart_to_VNB(r0::SVector{3, Float64}, v0::SVector{3, Float64})
    v_hat = v0 / norm(v0)
    n = cross(r0, v0)
    n_hat = n / norm(n)
    b_hat = cross(v_hat, n_hat)

    R = SMatrix{3, 3}(hcat(v_hat, n_hat, b_hat))'

    return R
end

Cart_to_VNB(u0::SVector{6, Float64}) = Cart_to_VNB(SVector{3, Float64}(u0[1:3]), SVector{3, Float64}(u0[4:6]))

@inline function Cart_to_RTN(r0::SVector{3, Float64}, v0::SVector{3, Float64})
    r_hat = r0 / norm(r0)
    n = cross(r0, v0)
    n_hat = n / norm(n)
    t_hat = cross(n_hat, r_hat)

    R = SMatrix{3, 3}(hcat(r_hat, t_hat, n_hat))'

    return R
end

Cart_to_RTN(u0::SVector{6, Float64}) = Cart_to_RTN(SVector{3, Float64}(u0[1:3]), SVector{3, Float64}(u0[4:6]))

@inline function Cart_to_Kep(u0::SVector{6,Float64}, μ::Float64)
    r = SVector{3,Float64}(u0[1], u0[2], u0[3])
    v = SVector{3,Float64}(u0[4], u0[5], u0[6])

    r_norm = norm(r)
    v2 = dot(v, v)

    h_vec = cross(r, v)
    h_norm = norm(h_vec)

    n_vec = SVector{3,Float64}(-h_vec[2], h_vec[1], 0.0)
    n_norm = norm(n_vec)

    e_vec = (1.0/μ) * ((v2 - μ/r_norm) * r - dot(r, v) * v)
    e = norm(e_vec)

    energy = 0.5*v2 - μ/r_norm
    a = -μ / (2.0*energy)

    i = acos(clamp(h_vec[3]/h_norm, -1.0, 1.0))

    # RAAN
    Ω = n_norm < 1e-12 ? 0.0 : acos(clamp(n_vec[1]/n_norm, -1.0, 1.0))
    if n_norm >= 1e-12 && n_vec[2] < 0.0
        Ω = 2π - Ω
    end

    # Argument of periapsis
    ω = (n_norm < 1e-12 || e < 1e-12) ? 0.0 : acos(clamp(dot(n_vec, e_vec)/(n_norm*e), -1.0, 1.0))
    if n_norm >= 1e-12 && e >= 1e-12 && e_vec[3] < 0.0
        ω = 2π - ω
    end

    # True anomaly
    θ = e < 1e-12 ? 0.0 : acos(clamp(dot(e_vec, r)/(e*r_norm), -1.0, 1.0))
    if e >= 1e-12 && dot(r, v) < 0.0
        θ = 2π - θ
    end

    return SVector{6,Float64}(a, e, Ω, i, ω, θ)
end

end # module post_utils