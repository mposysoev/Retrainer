module Retrainer

include("config_loader.jl")
include("entities.jl")
include("model_loader.jl")
include("model_initializer.jl")
include("logger.jl")
include("model_utils.jl")
include("training.jl")
include("plotter.jl")
include("model_saver.jl")
include("model_tester.jl")

function main()
    input_file_name = "input.toml"
    student_params, training_params, testing_params, teacher_file_path,
    student_file_path = parse_input_file(input_file_name)

    teacher_model = load_model(teacher_file_path)
    student_model = init_model(student_params)

    teacher_params = init_params_from_model(teacher_model)

    hello_message(input_file_name, teacher_file_path, student_file_path,
        teacher_params, student_params, training_params, testing_params)

    student_model, losses = train(
        teacher_model, student_model, teacher_params, student_params, training_params)

    compute_model_accuracy(
        student_model, teacher_model, student_params, teacher_params, testing_params)

    plot_model_parameters(teacher_model)
    plot_model_parameters(student_model)
    plot_loss_function(losses)

    save_model(student_model, student_file_path)
end
end # module Retrainer
