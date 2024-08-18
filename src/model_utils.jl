include("entities.jl")

using Flux

function get_network_structure(model::Flux.Chain)::Vector{Int64}
    structure = []
    for layer in model.layers
        push!(structure, size(layer.weight)[2])
    end
    push!(structure, size(model.layers[end].weight)[1])

    structure = convert(Vector{Int64}, structure)
    return structure
end

function get_network_activations(model::Flux.Chain)::Vector{String}
    activation_functions = String[]

    for layer in model.layers
        if hasproperty(layer, :σ)
            push!(activation_functions, string(layer.σ))
        else
            push!(activation_functions, "unknown")
        end
    end
    return activation_functions
end

function check_bias_usage(model::Flux.Chain)::Bool
    for layer in model.layers
        if !hasproperty(layer, :bias) || layer.bias === nothing
            return false
        end
    end
    return true
end

function get_weights_precision(model::Flux.Chain)::String
    for layer in model.layers
        if hasproperty(layer, :weight)
            weight_type = eltype(layer.weight)
            if weight_type == Float64
                return "f64"
            elseif weight_type == Float32
                return "f32"
            elseif weight_type == Float16
                return "f16"
            else
                return "unknown"
            end
        end
    end
    return "unknown"
end

function init_params_from_model(model::Flux.Chain)::NeuralNetParams
    structure = get_network_structure(model)
    activations = get_network_activations(model)
    use_bias = check_bias_usage(model)
    precision = get_weights_precision(model)
    return NeuralNetParams(structure, activations, use_bias, precision)
end
