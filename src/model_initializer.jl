include("entities.jl")

using Flux

function init_model(params::NeuralNetParams)::Flux.Chain
    structure = params.structure
    activations = params.activations
    use_bias = params.use_bias
    precision = params.precision

    # Словарь с функциями активации
    activation_dict = Dict(
        "relu" => relu,
        "σ" => σ,
        "tanh" => tanh,
        "identity" => identity
    )

    layers = []
    for i in 1:length(activations)

        # Получаем функцию активации из словаря
        activation_func = get(activation_dict, activations[i], identity)

        push!(
            layers,
            Dense(
                structure[i],
                structure[i + 1],
                activation_func,
                bias = use_bias
            )
        )
    end

    model = Chain(layers...)

    if precision == "f64"
        model = f64(model)
    elseif precision == "f32"
        model = f32(model)
    elseif precision == "f16"
        model = f16(model)
    end

    return model
end
