import torch
import torch.nn as nn

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.char_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentence, lengths):
        embeds = self.char_embeddings(sentence)
        lengths = lengths.reshape(-1)
        embeds_pack = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True)
        lstm_pack_out, _ = self.lstm(embeds_pack)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_pack_out, batch_first=True)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space