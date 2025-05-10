import torch
import torch.nn as nn
import lightning as L
from module.model import CBOW

class Train_model(L.LightningModule):
    def __init__(
        self,
        vocab_size,
        word_probs,
        device='cpu',
        batch_size=32,
        embedding_dim=100,
        context_size=3,
        num_negative=5,
        learning_rate=0.001
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.word_probs = word_probs
        self.my_device = device
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.num_negative = num_negative
        self.learning_rate = learning_rate
        self.model = CBOW(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            num_negative=self.num_negative
        )
        self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):

        context = batch['context']  #context[batch, context_size]
        target = batch['target']    #target[batch]

        # Generate negative samples
        negative_samples = torch.zeros((target.shape[0], self.num_negative), dtype=torch.long).to(self.my_device)
        for i in range(target.shape[0]):
            word_probs = self.word_probs.clone().detach()
            word_probs[target] = 0
            word_probs /= word_probs.sum()
            negative_samples[i] = torch.multinomial(word_probs, self.num_negative, replacement=True)

        # Forward pass
        pred = self.model(context, target.view(-1, 1), negative_samples)

        # loss calculation
        labels = torch.cat((torch.ones(target.shape[0], 1), torch.zeros(target.shape[0], self.num_negative)), dim=1).to(self.my_device)
        # pred = nn.Sigmoid(pred)     # self.criterion = nn.CrossEntropyLoss()
        loss = self.criterion(pred, labels)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer