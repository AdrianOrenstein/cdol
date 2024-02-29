from tinygrad.tensor import Tensor as tn
from tinygrad.nn import Linear  


class MLP:
    def __init__(self, input_dim, num_hidden_layers, embed_dim, output_dim):
        self.input_layer = Linear(input_dim, embed_dim)
        
        self.hidden_layers = []
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(Linear(embed_dim, embed_dim))
            
        self.output_layer = Linear(embed_dim, output_dim)
            

    def forward(self, x):       
        features = self.input_layer(x)
        for hidden_layer in self.hidden_layers:
            features = hidden_layer(features)
            
        features = self.output_layer(features)
        return features

    def __call__(self, x):
        return self.forward(x)


if __name__ == "__main__":
    mlp = MLP(10, 2, 20, 5)
    x = tn.rand(10)
    y = mlp(x)

    print(y.numpy())
