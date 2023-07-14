from nn_without_frameworks import numpy_nn as nn


class Model(nn.Module):
    def __init__(self, n_states, n_actions, ):
        super(Model, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.fc1 = nn.layers.Dense(self.n_states,
                                   128,
                                   nn.acts.ReLU(),
                                   nn.initializers.HeNormal(nn.acts.ReLU())
                                   )
        self.fc2 = nn.layers.Dense(128,
                                   256,
                                   nn.acts.ReLU(),
                                   nn.initializers.HeNormal(nn.acts.ReLU())
                                   )
        self.q_values = nn.layers.Dense(256,
                                        self.n_actions,
                                        weight_initializer=nn.initializers.XavierUniform()
                                        )

    def forward(self, inputs, eval=False):
        x = inputs
        x = self.fc1(x)
        x = self.fc2(x)
        return self.q_values(x)
