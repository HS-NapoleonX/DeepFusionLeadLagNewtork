
import torch
import torch.nn as nn

str2activation = {"relu" : nn.ReLU, "sigmoid" : nn.Sigmoid, "none" : nn.Identity}

class mlp(nn.Module):

    def __init__(self, input_dim, output_dims, activation="relu", last_activation="none", dropout=0.2):
        """
        input_dim (int)
        outputs_dims (list(int))
        activation (str)
        last_activation (str)
        """
        super().__init__()
        dims = [input_dim] + output_dims
        activation_layer = str2activation[activation]
        self.blocs = nn.ModuleList([nn.Sequential(nn.Linear(in_features=dims[i], out_features=dims[i+1]),
                                                  activation_layer(),
                                                  nn.Dropout(p=dropout)) for i in range(len(output_dims) - 1)])
        output_activation_layer = str2activation[last_activation]
        self.output_layer = nn.Sequential(nn.Linear(in_features=dims[-2], out_features=dims[-1]),
                                          output_activation_layer())

    def forward(self, input):
        out = input
        for bloc in self.blocs:
            out = bloc(input)
        out = self.output_layer(out)
        return out

class mda(nn.Module):

    def __init__(self, N, M, input_dims, dims, activation="relu", dropout=0.2):
        """
        N (int) : number of input nets
        M (int) : number of layers for each individual encoder / decoder
        dims (list(int))
        activation (str)
        dropout (float)
        """
        super().__init__()

        self.N = N
        self.split_size = dims[M-1]

        self.input_encoders = nn.ModuleList([mlp(input_dim=input_dims[i],
                                                 output_dims=dims[:M],
                                                 activation=activation,
                                                 dropout=dropout) for i in range(N)
                                            ])

        self.encoder = mlp(input_dim=self.N * dims[M-1],
                           output_dims=dims[M:],
                           activation=activation,
                           dropout=dropout)


        self.decoder = mlp(input_dim=dims[-1],
                           output_dims=list([self.N * dims[M-1]] + dims[M:-1])[::-1],
                           activation=activation,
                           dropout=dropout)

        self.output_decoders = nn.ModuleList([mlp(input_dim=dims[M-1],
                                                  output_dims=([input_dims[i]] + dims[:M-1])[::-1],
                                                  activation=activation,
                                                  last_activation="relu",
                                                  dropout=dropout) for i in range(N)
                                            ])

    def encode(self, input):
        out = [self.input_encoders[i](input[:, i, :]) for i in range(self.N)]
        out = torch.cat(out, dim=1)
        out = self.encoder(out)
        return out

    def forward(self, input):
        out = self.encode(input)
        out = self.decoder(out)
        out = list(torch.split(out, self.split_size, dim=1))
        out = [self.output_decoders[i](out[i]).unsqueeze(1) for i in range(self.N)]
        out = torch.cat(out, dim=1)
        return out

if __name__ == "__main__":

    import numpy as np
    import torch

    sample = np.zeros((1, 7, 10)).astype(np.float32)
    sample = torch.from_numpy(sample)

    input_dims = [10] * 7
    dims = [5, 2, 10, 5]

    model = mda(7, 2, input_dims, dims)
    print(model)
    out = model(sample)
    print(out)
    print(out.shape)

    print(model.encode(sample))
    print(model.encode(sample).shape)
