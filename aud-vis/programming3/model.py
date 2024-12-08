import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import util


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super().__init__()
        pe = None
        ################################################################################
        # TODO: compute the positional encoding                                        #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * denominator)
        pe[:, 1::2] = torch.cos(position * denominator)
        pe = pe.unsqueeze(0) # size = (batch_size, max_len, d_model (feature_size))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return x


class HarryPotterTransformer(nn.Module):
    def __init__(self, vocab_size, feature_size, num_heads):
        super(HarryPotterTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.feature_size = feature_size
        self.num_heads = num_heads
        self.best_accuracy = -1

        self.embedding = None
        self.transformer_encoder = None
        self.decoder = None
        self.pos_encoding = None # you can omit this for Task 4

        ################################################################################
        # TODO: define the network                                                     #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.embedding = nn.Embedding(vocab_size, feature_size)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dim_feedforward=4 * feature_size, dropout=0.1), num_layers=2)
        self.decoder = nn.Linear(feature_size, vocab_size)

        self.pos_encoding = PositionalEncoding(feature_size)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

    def forward(self, x):        
        attn_mask = None # you can omit this for Task 4 and Task 5
        ################################################################################
        # TODO: finish the forward pass                                                #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        '''
        x: Tensor, shape [batch_size, sequence_length]
        '''

        # # task 3
        # x = self.embedding(x) # size = (batch_size, seq_len, feature_size)
        # x = x.permute(1, 0, 2) # size = (seq_len, batch_size, feature_size)
        # x = self.transformer_encoder(x) # size = (seq_len, batch_size, feature_size)
        # x = x.permute(1, 0, 2) # size = (batch_size, seq_len, feature_size)
        # x = self.decoder(x) # size = (batch_size, seq_len, feature_size)

        # # task 4
        # x = self.embedding(x) # size = (batch_size, seq_len, feature_size)
        # x = self.pos_encoding(x)
        # x = x.permute(1, 0, 2) # size = (seq_len, batch_size, feature_size)
        # x = self.transformer_encoder(x) # size = (seq_len, batch_size, feature_size)
        # x = x.permute(1, 0, 2) # size = (batch_size, seq_len, feature_size)
        # x = self.decoder(x) # size = (batch_size, seq_len, feature_size)

        # task 5
        attn_mask = torch.full((x.size(1), x.size(1)), float('-inf'))
        attn_mask = torch.triu(attn_mask, diagonal=1) # only keep the upper triangle
        x = self.embedding(x) # size = (batch_size, seq_len, feature_size)
        x = self.pos_encoding(x)
        x = x.permute(1, 0, 2) # size = (seq_len, batch_size, feature_size)
        x = self.transformer_encoder(x, mask=attn_mask) # size = (seq_len, batch_size, feature_size)
        x = x.permute(1, 0, 2) # size = (batch_size, seq_len, feature_size)
        x = self.decoder(x) # size = (batch_size, seq_len, feature_size)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

        return x

    # This defines the function that gives a probability distribution and implements the temperature computation.
    def inference(self, x, temperature=1):
        x = x.view(1, -1)
        x = self.forward(x)
        x = x[0][-1].view(1, -1)
        x = x / max(temperature, 1e-20)
        x = F.softmax(x, dim=1)
        return x
    
    # Predefined loss function
    def loss(self, prediction, label, reduction='mean'):
        loss_val = F.cross_entropy(prediction.view(-1, self.vocab_size), label.view(-1), reduction=reduction)
        return loss_val

    # Saves the current model
    def save_model(self, file_path, num_to_keep=1):
        util.save(self, file_path, num_to_keep)

    # Saves the best model so far
    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        if accuracy > self.best_accuracy:
            self.save_model(file_path, num_to_keep)
            self.best_accuracy = accuracy

    def load_model(self, file_path):
        util.restore(self, file_path)

    def load_last_model(self, dir_path):
        return util.restore_latest(self, dir_path)