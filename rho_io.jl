module rho_io

using TOML, DelimitedFiles

function next_index(data_dir::AbstractString)
    indices = Int[]
    for name in readdir(data_dir)
        matched = match(r"^rho_evo_(\d+)\.csv$", name)
        isnothing(matched) || push!(indices, parse(Int, matched[1]))
    end
    return isempty(indices) ? 0 : maximum(indices) + 1
end

"""Save metadata and diagnostics to the next numbered CSV shared by both simulations."""
function save_evolution(data_dir::AbstractString, metadata, data, plot_ψ::Bool; index::Union{Nothing,Int}=nothing)
    mkpath(data_dir)
    index = isnothing(index) ? next_index(data_dir) : index
    path = joinpath(data_dir, "rho_evo_$(index).csv")
    # Exclusive creation also prevents overwriting if another run finishes concurrently.
    flags = Base.JL_O_WRONLY | Base.JL_O_CREAT | Base.JL_O_EXCL
    io = Base.Filesystem.open(path, flags, 0o666)
    try
        config = sprint(buffer -> TOML.print(buffer, metadata))
        for line in split(chomp(config), '\n')
            println(io, isempty(line) ? "#" : "# " * line)
        end
        println(io, plot_ψ ? "time_s,rho_VB,rho_RT,psi_BR" : "time_s,rho_VB,rho_RT")
        writedlm(io, data, ',')
    finally
        close(io)
    end
    return path
end

end # module rho_io
