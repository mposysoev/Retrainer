include("entities.jl")

using Random
using Flux

function train_different_input_size(
        teacher::Flux.Chain, student::Flux.Chain, teacher_params::NeuralNetParams,
        student_params::NeuralNetParams, params::TrainingParams)
    opt_state = Flux.setup(Flux.Adam(params.learning_rate), student)
    losses = Vector{Float64}(undef, params.epochs)
    coef = params.input_scale_coef
    input_size = maximum([teacher_params.structure[1], student_params.structure[1]])
    random_input = Vector{Float64}(undef, input_size)

    for epoch in 1:(params.epochs)
        rand!(random_input)
        random_input .*= params.input_scale_coef
        teacher_input = @view random_input[1:teacher_params.structure[1]]
        student_input = @view random_input[1:student_params.structure[1]]

        target_output = teacher(teacher_input)

        loss, grads = Flux.withgradient(student) do m
            student_output = m(student_input)
            Flux.mse(student_output, target_output)
        end
        Flux.update!(opt_state, student, grads[1])
        losses[epoch] = loss

        if epoch % (params.epochs // 10) == 0
            print("\rEpoch $epoch: loss_mse = $(loss)")
        end
    end

    return student, losses
end

function train(
        teacher::Flux.Chain, student::Flux.Chain, teacher_params::NeuralNetParams,
        student_params::NeuralNetParams, params::TrainingParams)
    opt_state = Flux.setup(Flux.Adam(params.learning_rate), student)
    losses = zeros(Float64, params.epochs)
    coef = params.input_scale_coef

    for epoch in 1:(params.epochs)
        random_input = coef .* rand(Float64, teacher_params.structure[1])
        target_output = teacher(random_input)
        loss, grads = Flux.withgradient(student) do m
            student_output = m(random_input)
            Flux.mse(student_output, target_output)
        end
        Flux.update!(opt_state, student, grads[1])
        losses[epoch] = loss
    end

    return student, losses
end
