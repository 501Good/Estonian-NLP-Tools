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
from tqdm import tqdm
from model import LSTMTagger
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EstTokenizer:
    
    def __init__(self, PATH):
        self.TEXT = pickle.load(open('TEXT.pkl', 'rb'))
        self.LABELS = pickle.load(open('LABELS.pkl', 'rb'))
        self.BATCH_SIZE = 1
        self.INPUT_DIM = len(self.TEXT.vocab)
        self.EMBEDDING_DIM = 100
        self.HIDDEN_DIM = 256
        self.OUTPUT_DIM = len(self.LABELS.vocab)
        
        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(device)
        
        self.PATH = PATH
        self.model = LSTMTagger(self.EMBEDDING_DIM, self.HIDDEN_DIM, self.INPUT_DIM, self.OUTPUT_DIM)
        self.model.load_state_dict(torch.load(PATH))
        self.model.to(device)
        self.model.eval()
        
        
    def __binary_accuracy(self, preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """

        #round predictions to the closest integer
        _, rounded_preds = torch.max(torch.sigmoid(preds), 1)
        correct = (rounded_preds == y).float() #convert into float for division 
        acc = correct.sum()/len(correct)
        return acc
    
    
    def evaluate(self, iterator):

        epoch_loss = 0
        epoch_acc = 0

        self.model.eval()

        with torch.no_grad():

            for batch in iterator:

                t, l = batch.text

                predictions = self.model(t, l)            
                #predictions = torch.argmax(predictions, dim=2)            
                predictions = predictions.reshape(-1, predictions.size()[-1])            
                predictions = predictions.float()

                labels = batch.labels.reshape(-1)
                labels = labels.long()

                loss = self.criterion(predictions, labels)            
                acc = self.__binary_accuracy(predictions, labels)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
    
    def tokenize(self, text, output='conllu'):
        text = [t for t in text.split("\n") if len(t) > 0]
        examples = [data.Example().fromlist([t], fields=[('text', self.TEXT)]) for t in text]
        dataset = data.Dataset(examples, fields=[('text', self.TEXT)])
        data_iter = data.BucketIterator(dataset, 
            batch_size=self.BATCH_SIZE,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            shuffle=False,
            device=device)
        
        with torch.no_grad():
            preds = []
            for batch in data_iter:
                t, l = batch.text
                predictions = self.model(t, l)           
                predictions = predictions.float()
                _, rounded_preds = torch.max(torch.sigmoid(predictions), 2)
                preds.append(rounded_preds)

        sents = []
        tokens = []
        for item in list(zip(text, preds[::-1])):
            text = item[0]
            tags = item[1]
            token = ''
            for i in tqdm(range(len(tags[0]))):
                if int(tags[0][i]) == 0:
                    token += text[i]
                elif int(tags[0][i]) == 1:
                    token += text[i]
                    if output == 'conllu':
                        space_after = 1 if text[i + 1] == ' ' else 0
                        tokens.append((token.strip(), space_after))
                    else:
                        tokens.append(token.strip())
                    token = ''
                else:
                    token += text[i]
                    if output == 'conllu':
                        tokens.append((token.strip(), 0))
                    else:
                        tokens.append(token.strip())
                    token = ''
                    sents.append(tokens)
                    tokens = []
        return sents


    def write_conllu(self, sents, filename='lstm_tokenizer_output.conllu'):
        with open(filename, 'w', encoding='utf-8') as f:
            for s_id, sent in enumerate(sents):
                sent_text = ''
                token_lines = []
                for i, token_info in enumerate(sent):
                    token, space_after = token_info[0], token_info[1]
                    if space_after == 1:
                        sent_text += token + ' '
                        token_line = '{}\t{}\t_\t_\t_\t_\t_\t_\t_\t_'.format(i + 1, token)
                        token_lines.append(token_line)
                    else:
                        sent_text += token
                        token_line = '{}\t{}\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No'.format(i + 1, token)
                        token_lines.append(token_line)
                f.write('# sent_id = {}\n'.format(s_id + 1))
                f.write('# text = {}\n'.format(sent_text))
                f.write('\n'.join(token_lines))
                f.write('\n\n')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str)
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--format', type=str, default='conllu', dest='format')

    args = parser.parse_args()

    model = args.model
    input_file = args.input
    output_file = args.output
    file_format = args.format

    est_tokenizer = EstTokenizer(model)

    test = open(input_file, encoding='utf-8').read()
    sents = est_tokenizer.tokenize(test)

    est_tokenizer.write_conllu(sents, filename=output_file)


if __name__ == '__main__':
    main()