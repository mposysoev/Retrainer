include("entities.jl")

using Statistics
using Random

function train(
        teacher::Flux.Chain, student::Flux.Chain, teacher_params::NeuralNetParams,
        student_params::NeuralNetParams, params::TrainingParams)
    opt = Flux.setup(Flux.Adam(params.learning_rate), student)
    losses = Vector{Float64}(undef, params.epochs)

    input_size = maximum([teacher_params.structure[1], student_params.structure[1]])
    random_input = Vector{Float64}(undef, input_size)

    loss(model, x, y) = mean(abs2.(model(x) .- y))

    for i in 1:(params.epochs)
        rand!(random_input)
        random_input .*= params.input_scale_coef
        teacher_input = @view random_input[1:teacher_params.structure[1]]
        student_input = @view random_input[1:student_params.structure[1]]

        target_output = teacher(teacher_input)

        data = [(student_input, target_output)]
        Flux.train!(loss, student, data, opt)

        loss_value = loss(
            student, student_input, target_output)
        losses[i] = loss_value

        if i % (params.epochs // 10) == 0
            print("\rIteration $i: loss = $(loss_value)")
        end
    end

    return student, losses
end

function train_old(
        teacher::Flux.Chain, student::Flux.Chain, teacher_params::NeuralNetParams,
        student_params::NeuralNetParams, params::TrainingParams)
    opt = Flux.setup(Flux.Adam(params.learning_rate), student)
    losses = Vector{Float64}(undef, params.epochs)
    coef = params.input_scale_coef
    for i in 1:(params.epochs)
        random_input = coef .* rand(Float64, teacher_params.structure[1])
        target_output = teacher(random_input)

        loss(model, x, y) = mean(abs2.(model(x) .- y))
        loss_value = loss(student, random_input, target_output)
        losses[i] = loss_value

        data = [(random_input, target_output)]
        Flux.train!(loss, student, data, opt)

        if i % (params.epochs // 10) == 0
            print("\rIteration $i: loss = $(loss_value)")
        end
    end

    return student, losses
end
