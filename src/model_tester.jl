include("entities.jl")

using Flux
using Statistics

function compute_model_accuracy(
        student::Flux.Chain, teacher::Flux.Chain, student_params::NeuralNetParams,
        teacher_params::NeuralNetParams, params::TestingParams)
    input_size = maximum([teacher_params.structure[1], student_params.structure[1]])

    difference = Vector{Float64}(undef, params.samples_num)
    for i in 1:(params.samples_num)
        random_input = params.input_scale_coef .* rand(Float64, input_size)
        teacher_output = teacher(random_input[1:teacher_params.structure[1]])
        student_output = student(random_input[1:student_params.structure[1]])
        difference[i] = abs((teacher_output[1] - student_output[1]) / teacher_output[1])
    end

    mean_error = Statistics.mean(difference) * 100
    std_error = Statistics.std(difference) * 100

    println()
    println("Student model error: $(round(mean_error, digits=2))% ± $(round(std_error, digits=2))%")
end
