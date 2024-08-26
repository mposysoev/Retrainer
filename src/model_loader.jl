using BSON: @load
using Flux

function load_model(file_path::AbstractString)::Flux.Chain
    model = nothing
    @load file_path model
    return model
end
