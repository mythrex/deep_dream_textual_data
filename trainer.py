import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from model import TextClassifier
import argparse
import torch
from torchtext import data
import random
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 2020
# Torch
torch.manual_seed(SEED)
# Cuda algorithms
torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='Short sample app')

parser.add_argument('-data_path',
                    '--data_path',
                    help='Enter Data Path',
                    required=True)
parser.add_argument('-embedding_dim',
                    '--embedding_dim',
                    help='Embedding Dimensions',
                    default=100)
parser.add_argument('-dropout',
                    '--dropout',
                    help='Dropout(Drop Probability)',
                    default=0.5)
parser.add_argument('-lr',
                    '--lr',
                    help='Learning Rate',
                    default=5e-4)
parser.add_argument('-batch_size',
                    '--batch_size',
                    help='Batch Size',
                    default=256)
parser.add_argument('-epochs',
                    '--epochs',
                    help='epochs',
                    default=50)

parser.add_argument('-gpus',
                    '--gpus',
                    help='Number of gpus',
                    default=0)
parser.add_argument('-progress_bar_refresh_rate',
                    '--progress_bar_refresh_rate',
                    help='Progress Bar refresh rate of wandb',
                    default=25)

parser.add_argument('-wandb_log_step',
                    '--wandb_log_step',
                    help='After how many steps need to log for train loop',
                    default=10)
parser.add_argument('-wandb_run_name',
                    '--wandb_run_name',
                    help='Name of Wandb Run',
                    default='run')
parser.add_argument('-wandb_project_name',
                    '--wandb_project_name',
                    help='Wandb Project Name',
                    default='deep_dream')
parser.add_argument('-model_ckpt_path',
                    '--model_ckpt_path',
                    help='Model Checkpoint Path',
                    default='./ckpts/model.ckpt')

args = parser.parse_args()
args = vars(args)
# print(args)

if __name__ == '__main__':
    TEXT = data.Field(
        tokenize='spacy', batch_first=True, include_lengths=True)

    LABEL = data.Field(batch_first=True, sequential=False)

    fields = [('text', TEXT), ('label', LABEL)]

    print(f"Loading file: {args['data_path']}")
    training_data = data.TabularDataset(
        path=args['data_path'], format='csv', fields=fields, skip_header=True)

    print("Splitting the data!")
    train_data, valid_data = training_data.split(
        split_ratio=0.7, random_state=random.seed(SEED))

    print("Building Vocab!")
    TEXT.build_vocab(train_data, min_freq=3, vectors="glove.6B.100d")
    LABEL.build_vocab(train_data)

    # #Load an iterator
    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size=args['batch_size'],
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        device=device)

    # print preprocessed text
    # print(vars(training_data.examples[0]))

    wandb_logger = WandbLogger(name=args['wandb_run_name'],
                               project=args['wandb_project_name'])
    wandb_logger.log_hyperparams(args)

    model = TextClassifier(args,
                           TEXT=TEXT,
                           LABEL=LABEL,
                           train_iterator=train_iterator,
                           valid_iterator=valid_iterator,
                           wandb_logger=wandb_logger)
    # wandb.watch(model)

    trainer = pl.Trainer(gpus=int(args['gpus']),
                         progress_bar_refresh_rate=args['progress_bar_refresh_rate'],
                         max_epochs=args['epochs'],
                         logger=[wandb_logger],
                         early_stop_callback=True)
    trainer.fit(model)

    ckpt_base_path = os.path.dirname(args['model_ckpt_path'])
    os.makedirs(ckpt_base_path, exist_ok=True)

    trainer.save_checkpoint(args['model_ckpt_path'])
    wandb.save(args['model_ckpt_path'])
