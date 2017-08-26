import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TreeLM(nn.Module):
    
    def __init__(self, args, attr_size, node_size):
        super(TreeLM, self).__init__()

        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        self.attr_size = attr_size
        self.node_size = node_size

        self.embedding_dim = args.embedding_dim
        self.layer_num = args.layer_num
        self.dropout_prob = args.dropout_prob
        self.lr = args.lr

        self.attr_embedding = nn.Embedding(self.attr_size, self.embedding_dim)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.lstm = nn.LSTM(input_size = self.embedding_dim,
                            hidden_size = self.embedding_dim,
                            num_layers= self.layer_num,
                            dropout = self.dropout_prob)

        self.fc = nn.Linear(self.embedding_dim, self.node_size)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.node_mapping = node_mapping

    def init_weights(self):
        """
            initialize all trainable weights
        """
        init_range = 0.1
        self.attr_embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size=None):
        """
            Create an initial version of hidden state
        """
        if batch_size:
            self.batch_size = batch_size
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.layer_num, self.batch_size, self.embedding_dim).zero_()),
                Variable(weight.new(self.layer_num, self.batch_size, self.embedding_dim).zero_()))

    def forward(self, inputs, hidden):
        """
            Do one forward one, returning logits and hidden states
        """
        embeds = self.dropout(self.attr_embedding(inputs))
        if (len(embeds.data.size()) == 2):
            embeds.data.unsqueeze_(0) #TODO: Strange incompatiblility between CPU and GPU
        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(embeds, hidden)
        logits = self.fc(outputs.view(-1, self.embedding_dim))
        logits = logits.view(self.seq_length, self.batch_size, self.node_size)
        return logits, hidden