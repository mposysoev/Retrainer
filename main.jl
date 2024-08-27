using ArgParse

include("src/Retrainer.jl")

using .Retrainer

function main()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--file", "-f"
        help = "Input file in TOML format with parameters"
        arg_type = String
        required = true
    end

    parsed_args = parse_args(ARGS, s)
    input_file = parsed_args["file"]

    student_params, training_params, testing_params, teacher_file_path,
    student_file_path, fine_tuning_params = parse_input_file(input_file)

    if fine_tuning_params.use_fine_tuning == false
        student_model, teacher_model, losses = run_initial_training(
            student_params, training_params,
            testing_params, teacher_file_path, student_file_path)

        plot_loss_function(losses, "retraining_loss_function_plot.png", true)
        return
    end

    if fine_tuning_params.use_fine_tuning == true
        student_model, teacher_model, losses = run_fine_tuning(
            student_file_path, fine_tuning_params.file_name, teacher_file_path,
            training_params, testing_params)
        plot_loss_function(losses, "fine_tuning_loss_function_plot.png", true)
        return
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
