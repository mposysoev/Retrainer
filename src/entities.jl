struct NeuralNetParams
    structure::Vector{Int64}
    activations::Vector{String}
    use_bias::Bool
    precision::String
end

struct TrainingParams
    epochs::Int64
    learning_rate::Float64
    input_scale_coef::Float64
end

struct TestingParams
    input_scale_coef::Float64
    samples_num::Int64
end

struct FineTuningParams
    use_fine_tuning::Bool
    file_name::String
end
