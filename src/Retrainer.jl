module Retrainer

using Flux
using Dates

include("entities.jl")
include("config_loader.jl")
include("model_loader.jl")
include("model_initializer.jl")
include("logger.jl")
include("model_utils.jl")
include("training.jl")
include("plotter.jl")
include("model_saver.jl")
include("model_tester.jl")

export run_initial_training, run_fine_tuning, NeuralNetParams, TrainingParams,
       TestingParams, plot_model_parameters, plot_loss_function

function run_initial_training(
        student_params::NeuralNetParams, training_params::TrainingParams,
        testing_params::TestingParams, teacher_file_path::String,
        student_file_path::String)
    start_time = Dates.now()

    teacher_model = load_model(teacher_file_path)
    student_model = init_model(student_params)

    teacher_params = init_params_from_model(teacher_model)

    student_model, losses = train(
        teacher_model, student_model, teacher_params, student_params, training_params)

    compute_model_accuracy(
        student_model, teacher_model, student_params, teacher_params, testing_params)

    save_model(student_model, student_file_path)

    finish_time = Dates.now()
    elapsed_time = Dates.canonicalize(finish_time - start_time)
    println()
    println("- Elapsed time: ", elapsed_time)
    println()

    return student_model, teacher_model, losses
end

function run_fine_tuning(
        student_file_path::AbstractString, output_file_name::AbstractString,
        teacher_file_path::AbstractString,
        training_params::TrainingParams, testing_params::TestingParams)
    start_time = Dates.now()

    teacher_model = load_model(teacher_file_path)
    student_model = load_model(student_file_path)

    teacher_params = init_params_from_model(teacher_model)
    student_params = init_params_from_model(student_model)

    student_model, losses = train(
        teacher_model, student_model, teacher_params, student_params, training_params)

    compute_model_accuracy(
        student_model, teacher_model, student_params, teacher_params, testing_params)

    save_model(student_model, output_file_name)

    finish_time = Dates.now()
    elapsed_time = Dates.canonicalize(finish_time - start_time)
    println()
    println("- Elapsed time: ", elapsed_time)
    println()

    return student_model, teacher_model, losses
end

end # module Retrainer
