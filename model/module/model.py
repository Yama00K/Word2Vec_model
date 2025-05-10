import torch
import torch.nn as nn
import lightning as L

class CBOW(nn.Module):
    def __init__(self, vocab_size, num_negative, embedding_dim=100):
        super().__init__()
        self.embedding_w1 = torch.nn.Embedding(vocab_size, embedding_dim)
        self.embedding_w2 = torch.nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_negative = num_negative

    def forward(self, context, target, negative_samples):
        embed_w1 = self.embedding_w1(context)   # [batch, context_size, embedding_dim]
        h = torch.mean(embed_w1, dim=1)      # [batch, embedding_dim]
        samples = torch.cat((target, negative_samples), dim=1)      # [batch, 1+num_negative]
        embed_w2 = self.embedding_w2(samples)       # [batch, embedding_dim]
        h_expanded = h.unsqueeze(1).expand(-1, 1+self.num_negative, -1)     # [batch, 1+num_negative, embedding_dim]
        pred = torch.sum(h_expanded * embed_w2, dim=2)   # [batch, 1+num_negative]
        return pred