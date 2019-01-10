import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchtext import data
from torch import autograd
from torch import tensor
import sys, traceback
import pickle
import numpy as np
import argparse
from model import LSTMTagger


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            print(self.best)
            return False

        if np.isnan(metrics):
            print('nan')
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            print('Improvement from {} to {}'.format(self.best, metrics))
            self.best = metrics
        else:
            print('No improvement')
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)



def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    _, rounded_preds = torch.max(torch.sigmoid(preds), 1)
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        t, l = batch.text
        
        predictions = model(t, l)
        #predictions = torch.argmax(predictions, dim=2)        
        predictions = predictions.reshape(-1, predictions.size()[-1])        
        predictions = predictions.float()
            
        labels = batch.labels.reshape(-1)      
        labels = labels.long()
        

        loss = criterion(predictions, labels)        
        acc = binary_accuracy(predictions, labels) 
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:
            
            t, l = batch.text

            predictions = model(t, l)            
            #predictions = torch.argmax(predictions, dim=2)            
            predictions = predictions.reshape(-1, predictions.size()[-1])            
            predictions = predictions.float()
            
            labels = batch.labels.reshape(-1)
            labels = labels.long()
            
            loss = criterion(predictions, labels)            
            acc = binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train_model(data_folder, patience, max_epoch, model_path):
    TEXT = data.Field(tokenize=list, include_lengths=True, batch_first=True)
    LABELS = data.Field(dtype=torch.float, tokenize=list, pad_token=None, unk_token=None, batch_first=True)

    train_data, val_data, test_data = data.TabularDataset.splits(
        path='data_folder', train='_train.tsv',
        validation='_dev.tsv', test='_test.tsv', format='tsv',
        fields=[('text', TEXT), ('labels', LABELS)], csv_reader_params={"quotechar": '|'})

    TEXT.build_vocab(train_data)
    LABELS.build_vocab(train_data)

    pickle.dump(TEXT, open('TEXT.pkl', 'wb'))
    pickle.dump(LABELS, open('LABELS.pkl', 'wb'))

    BATCH_SIZE = 64
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        device=device)

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = len(LABELS.vocab)

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=patience)

    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = max_epoch

    train_losses = []
    val_losses = []

    for epoch in range(N_EPOCHS):

        try:
            train_loss, train_acc = train(model, train_iter, optimizer, criterion)
            train_losses.append(train_loss)
        except (TypeError, ValueError):
            print("Exception in user code:")
            print("-"*60)
            traceback.print_exc(file=sys.stdout)
            print("-"*60)
            break
        valid_loss, valid_acc = evaluate(model, val_iter, criterion)
        val_losses.append(valid_loss)
        
        if early_stop.step(valid_loss):
            print('Stopped learning due to lack of progress.')
            break
        
        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')

    torch.save(model.state_dict(), model_path)

    test_loss, test_acc = evaluate(model, test_iter, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, action='store')
    parser.add_argument('model_path', type=str, action='store')
    parser.add_argument('--patience', type=int, default=5, action='store', dest='patience')
    parser.add_argument('--max_epoch', type=int, default=200, action='store', dest='max_epoch')

    args = parser.parse_args()

    data_folder = args.data_folder
    patience = args.patience 
    max_epoch = args.max_epoch
    model_path = args.model_path

    train_model(data_folder, patience, max_epoch, model_path)

if __name__ == '__main__':
    main()