include("entities.jl")

using TOML

using Flux

function parse_input_file(file_path = "input.toml")
    settings = TOML.parsefile(file_path)
    teacher_settings = settings["teacher_model"]
    student_settings = settings["student_model"]
    training_settings = settings["training"]
    testing_settings = settings["testing"]

    training_params = TrainingParams(
        training_settings["epochs"], training_settings["learning_rate"],
        training_settings["input_scale_coef"])

    testing_params = TestingParams(
        training_settings["input_scale_coef"], testing_settings["samples_num"])

    teacher_file_path = teacher_settings["file_path"]
    student_file_path = student_settings["file_name"]

    student_params = NeuralNetParams(
        student_settings["structure"], student_settings["activations"],
        student_settings["use_bias"], student_settings["precision"])

    return student_params, training_params, testing_params, teacher_file_path,
    student_file_path
end
