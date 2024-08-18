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

function plot_loss_function(losses::Vector{Float64}, console = false)
    p = plot(losses; yaxis = "loss", xaxis = (:log10, "epoch"),
        label = "Loss function", size = (1200, 800), fmt = :png)

    if console
        savefig(p, "loss-function-plot.png")
    else
        display(p)
    end
end
