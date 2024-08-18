using BSON: @save
using Flux

function save_model(model::Flux.Chain, file_path::AbstractString)
    @save file_path model
end
