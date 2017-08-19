#import torch
from torch.autograd import Variable
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim

class CharLM(nn.Module):
    """
        A Character Level Language Model
        Feed in a batched sequences with vocabulary
    """
    def __init__(self, args, mapping):
        super(CharLM, self).__init__()

        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        self.vocab_size = args.vocab_size
        self.embedding_dim = args.embedding_dim
        self.layer_num = args.layer_num
        self.dropout_prob = args.dropout_prob
        self.lr = args.lr
        self.char_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.lstm = nn.LSTM(input_size = self.embedding_dim,
                            hidden_size = self.embedding_dim,
                            num_layers= self.layer_num,
                            dropout = self.dropout_prob)

        self.fc = nn.Linear(self.embedding_dim, self.vocab_size)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.mapping = mapping

    def init_weights(self):
        """
            initialize all trainable weights
        """
        init_range = 0.1
        self.char_embedding.weight.data.uniform_(-init_range, init_range)
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
        embeds = self.dropout(self.char_embedding(inputs))
        outputs, hidden = self.lstm(embeds, hidden)
        logits = self.fc(outputs.view(-1, self.embedding_dim))
        logits = logits.view(self.seq_length, self.batch_size, self.vocab_size)
        return logits, hidden
