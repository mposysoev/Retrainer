using Plots
using Flux
using Base64

function plot_model_parameters(model::Flux.Chain)
    for layer in model
        # Check if the layer has parameters (weights and biases)
        if hasmethod(Flux.params, Tuple{typeof(layer)})
            layer_params = Flux.params(layer)
            for p in layer_params
                # Define a custom color gradient with white at 0
                color_gradient = cgrad([:blue, :white, :red], [0.0, 0.5, 1.0], rev = false)
                # Set color limits to ensure 0 is always white
                color_limits = (-1.0, 1.0)
                # Assuming the parameter is a 2D array (for weights)
                if ndims(p) == 2
                    x_ticks = 1:size(p, 2)
                    y_ticks = 1:size(p, 1)
                    p_plot = heatmap(
                        Array(p),
                        title = "Weights",
                        xticks = (x_ticks, string.(x_ticks)),
                        yticks = (y_ticks, string.(y_ticks)),
                        c = color_gradient,
                        clims = color_limits
                    )
                    display(p_plot)
                    # For biases or any 1D parameter, we convert them into a 2D array for the heatmap
                elseif ndims(p) == 1
                    x_ticks = 1:length(p)
                    p_plot = heatmap(
                        reshape(Array(p), 1, length(p)),
                        title = "Biases",
                        xticks = (x_ticks, string.(x_ticks)),
                        yticks = (1, "1"),
                        c = color_gradient,
                        clims = color_limits
                    )
                    display(p_plot)
                end
            end
        end
    end
end

function plot_loss_function(losses::Vector{Float64}, filename = "loss_function_plot.png", save_png = false)
    save = false

    if isempty(losses)
        error("Input vector 'losses' cannot be empty.")
    end

    if length(losses) > 1000000
        println("Plots.jl doesnt work in interactive mode for big Vectors.")
        println("Resulting plot would be saved in $filename")
        save = true
    end

    period = Int(round(sqrt(length(losses))))
    moving_average = [mean(losses[max(1, i - period + 1):i]) for i in 1:length(losses)]

    title = "Loss Function"
    p = plot(losses; xaxis = (:log10, "epoch"), yaxis = "loss",
        label = "Loss function", size = (800, 600), fmt = :png, title = title)

    plot!(p, moving_average; label = "Moving Average", linewidth = 2.0)
    if save || save_png
        savefig(p, filename)
        return
    end

    display(p)
end
