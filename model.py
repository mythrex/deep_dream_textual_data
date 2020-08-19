import torch.nn as nn
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
# handling text data
from torchtext import data
from torchtext.data import Field, BucketIterator, Batch
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.loggers import WandbLogger
# import wandb
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TextClassifier(pl.LightningModule):

    def __init__(self, hparams, TEXT, LABEL, train_iterator, valid_iterator, wandb_logger=None):
        super(TextClassifier, self).__init__()
        self.hparams = hparams
        self.TEXT = TEXT
        self.LABEL = LABEL
        self.train_iterator = train_iterator
        self.valid_iterator = valid_iterator
        self.wandb_logger = wandb_logger
        # model
        # embedding layer
        self.embedding = nn.Embedding(
            len(self.TEXT.vocab), hparams['embedding_dim'])
        # Initialize the pretrained embedding
        pretrained_embeddings = self.TEXT.vocab.vectors
        self.embedding.weight.data.copy_(pretrained_embeddings)
        # set embeddings non trainable
        self.embedding.weight.requires_grad = False
        self.bn1 = nn.BatchNorm1d(100)

        # dense layer
        self.fc1 = nn.Sequential(nn.Linear(100, 256),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(256),
                                 nn.Dropout(p=hparams['dropout']),
                                 )

        # dense layer
        self.fc2 = nn.Sequential(nn.Linear(256, 512),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(512),
                                 nn.Dropout(p=hparams['dropout']))

        self.fc3 = nn.Sequential(nn.Linear(512, len(self.LABEL.vocab)),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(len(self.LABEL.vocab)),
                                 nn.Dropout(p=hparams['dropout']))

    def forward(self, text):

        # text = [batch size,sent_length]
        embedded = self.embedding(text)
        embedded_avg = embedded.mean(1)
        embedded_avg = self.bn1(embedded_avg)
        # pdb.set_trace()

        # Final activation function
        fc1 = self.fc1(embedded_avg)
        fc2 = self.fc2(fc1)
        fc3 = self.fc3(fc2)
        return fc3, embedded, fc2

    def forward_from_embeddings(self, embedded):
        # text = [batch size,sent_length]

        # Final activation function
        fc1 = self.fc1[:2](embedded)
        fc2 = self.fc2[:2](fc1)
        fc3 = self.fc3[:2](fc2)
        return fc3, embedded, fc2

    def transfer_batch_to_device(self, batch, device):
        if isinstance(batch, Batch):
            pass
        else:
            batch = super().transfer_batch_to_device(batch, device)
        return batch

    def train_dataloader(self):
        return self.train_iterator

    def val_dataloader(self):
        return self.valid_iterator

    def loss_fn(self, logits, y):
        return F.cross_entropy(logits, y)

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams['lr'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.hparams['epochs'])
        return [self.optimizer], [self.scheduler]

    def training_step(self, train_batch, batch_idx):
        # retrieve text and no. of words
        text, text_lengths = train_batch.text

        logits, _, _ = self.forward(text)
        # convert to 1D tensor
        logits = logits.squeeze()
        # compute the loss
        loss = self.loss_fn(logits, train_batch.label)

        # compute the binary accuracy
        acc = accuracy(torch.argmax(logits, dim=-1), train_batch.label)

        cur_lr = self.get_lr(self.optimizer)
        logs = {
            'train_acc': acc,
            'train_loss': loss,
            'lr': cur_lr
        }
        # logging step
        if(batch_idx % self.hparams.get('wandb_log_step', 1) == 0 and self.wandb_logger):
            self.wandb_logger.log_metrics(logs)
        return {'loss': loss, 'logs': logs}

    def validation_step(self, val_batch, batch_idx):
        # retrieve text and no. of words
        text, text_lengths = val_batch.text

        logits, _, _ = self.forward(text)
        # convert to 1d tensor
        logits = logits.squeeze()

        # compute loss and accuracy
        loss = self.loss_fn(logits, val_batch.label)
        acc = accuracy(torch.round(logits), val_batch.label)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        bar = {'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc}
        return {'val_loss': avg_loss, 'progress_bar': bar, 'log': bar}
