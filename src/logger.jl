include("entities.jl")

function hello_message(
        input_file::String, teacher_file_path::String, student_file_path::String,
        input::NeuralNetParams, output::NeuralNetParams, training::TrainingParams, testing::TestingParams)
    println("""
/*****************************************************************************/
/                               Retrainer                                     /
/*****************************************************************************/
    """)
    println("Parameters read from file: $(input_file)")
    println()
    println("Neural Network would be changed from:")
    println("- File: $(teacher_file_path)")
    println("- Structure: $(input.structure)")
    println("- Activations: $(input.activations)")
    println("- Use bias: $(input.use_bias)")
    println("- Precision: $(input.precision)")
    println("To:")
    println("- Saved to file: $(student_file_path)")
    println("- Structure: $(output.structure)")
    println("- Activations: $(output.activations)")
    println("- Use bias: $(output.use_bias)")
    println("- Precision: $(output.precision)")
    println()
    println("Training Parameters:")
    println("- Epochs: $(training.epochs)")
    println("- Learning rate: $(training.learning_rate)")
    println("- Optimizer: Adam")
    println("- Input Scale Coefficient: $(training.input_scale_coef)")
    println()
    println("Testing Parameters:")
    println("- Input Scale Coefficient: $(testing.input_scale_coef)")
    println("- Samples number: $(testing.samples_num)")
    println("--------------------------------------------------------------------------------")
    println()
end
