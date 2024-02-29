from tinygrad.tensor import Tensor as tn


class Linear:
    def __init__(self, input_size, output_size):
        # Initialize weights and bias
        self.weights = tn.randn(input_size, output_size)
        self.bias = tn.randn(1, output_size)

        print(
            f"Initialized Linear layer with weights: {self.weights.shape} and bias: {self.bias.shape}"
        )

    def forward(self, inputs):
        # Perform dot product of inputs and weights
        weighted_sum = tn.dot(inputs, self.weights)
        print(f"\t-- {inputs.shape} * {self.weights.shape} = {weighted_sum.shape}")

        # Add the bias
        output = weighted_sum + self.bias
        print(f"\t-- {weighted_sum.shape} + {self.bias.shape} = {output.shape}")

        return output


if __name__ == "__main__":
    # Create a Linear layer
    linear = Linear(5, 3)

    # Create some random inputs
    inputs = tn.randn(1, 5)

    # Forward pass
    output = linear.forward(inputs)

    print(f"Output shape: {output.shape}, Output: {output.numpy()}")
