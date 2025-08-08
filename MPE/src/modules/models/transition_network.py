import torch
import torch.nn as nn
import torch.nn.functional as F


class TransitionNetwork(nn.Module):
    def __init__(self, input_shape, args):
        super(TransitionNetwork, self).__init__()
        self.args = args

        self.rnn = nn.GRU(
            input_size=args.rnn_hidden_dim, 
            hidden_size=args.rnn_hidden_dim, 
            batch_first=True
        )

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.args.batch_size * self.args.n_agents, self.args.rnn_hidden_dim).cuda()

    def forward(self, inputs, hidden_state):

        output, h = self.rnn(inputs, hidden_state)

        return h

