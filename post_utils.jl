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

Cart_to_VNB(u0::SVector{6, Float64}) = Cart_to_VNB(u0[1:3], u0[4:6])

end # module post_utils