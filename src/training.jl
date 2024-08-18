include("entities.jl")

using Statistics

function train(
        teacher::Flux.Chain, student::Flux.Chain, teacher_params::NeuralNetParams,
        student_params::NeuralNetParams, params::TrainingParams)
    opt = Flux.setup(Flux.Adam(params.learning_rate), student)
    losses = Vector{Float64}(undef, params.epochs)

    input_size = maximum([teacher_params.structure[1], student_params.structure[1]])

    loss(model, x, y) = mean(abs2.(model(x) .- y))

    for i in 1:(params.epochs)
        random_input = params.input_scale_coef .* rand(Float64, input_size)
        target_output = teacher(random_input[1:teacher_params.structure[1]]) # возможно ещё +1
        loss_value = loss(
            student, random_input[1:student_params.structure[1]], target_output)
        losses[i] = loss_value

        data = [(random_input[1:student_params.structure[1]], target_output)]
        Flux.train!(loss, student, data, opt)

        if i % (params.epochs // 10) == 0
            print("\rIteration $i: loss = $(loss_value)")
        end
    end

    return student, losses
end
