import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence
import numpy
import os
from tqdm import tqdm
import gc
import argparse
from sklearn.metrics import roc_auc_score

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class TextLSTMClassifier(nn.Module):
    def __init__(self, hidden_dim, lstm_hidden=128, bidirectional=True):
        super(TextLSTMClassifier, self).__init__()
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers = 1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden * 2 if bidirectional else lstm_hidden
        self.fc1 = nn.Linear(lstm_out_dim, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):  # x: (B, T, H), lengths: (B,)
        packed_out, (h_n, _) = self.lstm(x)

        # Use the final hidden state from both directions
        if self.bidirectional:
            final_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)  # shape: (B, 2 * lstm_hidden)
        else:
            final_hidden = h_n[-1]  # shape: (B, lstm_hidden)

        x = F.relu(self.fc1(final_hidden))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)
    

class TextDataset(Dataset):
    def __init__(self, punchline_list,full_sentence_list, labels):
        self.punchlines = punchline_list
        self.full_sentences = full_sentence_list
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        return{
            'punchline': self.punchlines[idx],
            'sentence': self.full_sentences[idx],
            'label': self.labels[idx]
        }
    
def collate_fn_text(batch):
    # already tensors so no need to convert them to tensor
    punchline_list = [item['punchline'] for item in batch]
    sentence_list = [item['sentence'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch],dtype=torch.float)

    padded_punchlines = pad_sequence(punchline_list,batch_first=True)
    padded_sentences = pad_sequence(sentence_list,batch_first=True)

    plengths = torch.tensor([v.shape[0] for v in punchline_list])
    slenghts = torch.tensor([v.shape[0] for v in sentence_list])

    return{
        'punchlines': padded_punchlines,
        'sentences': padded_sentences,
        'labels':labels,
        'punchline_lengths':plengths,
        'sentence_lengths':slenghts

    }

def load_pickle_data(pkls_dir,filter_json={}):
    full_resulting_dict = {}
    resulting_dict = {}
    # disabling for faster pickle load
    gc.disable()
    if len(filter_json)>0:
        pkl_files = [f for f in os.listdir(pkls_dir) if f.split('.')[0] in filter_json]
    else:
        pkl_files = [f for f in os.listdir(pkls_dir)]
    for filename in tqdm(pkl_files,desc='loading pickles',disable=True):
        file_path = os.path.join(pkls_dir,filename)
        with open(file_path,'rb') as f:
            obj = pickle.load(f)
            id = filename.split('.')[0]
            ctensor = obj['context_embedding']
            ptensor = obj['punchline_embedding']
            full_sentence = torch.cat([ctensor,ptensor])
            full_resulting_dict[id] = full_sentence
            resulting_dict[id] =ptensor
    gc.enable()
    return resulting_dict,full_resulting_dict

def accuracy(outputs,labels):
    preds = (outputs > 0.5).float()
    correct = preds.eq(labels).float().sum().item()
    return correct / labels.size(0)

def train(model,dataloader,optimizer,loss_fn,full=False):
    model.train()
    train_running_loss = 0.0
    train_running_acc = 0.0

    for data in tqdm(dataloader,desc='Training...',total=len(dataloader),disable=True):
        punchline,sentence,labels,plength,slength = data['punchlines'].to('cuda'),data['sentences'].to('cuda'),data['labels'].to('cuda'),data['punchline_lengths'],data['sentence_lengths']

        if full:
            input_feature = pack_padded_sequence(sentence,slength,batch_first=True,enforce_sorted=False)
        else:
            input_feature = pack_padded_sequence(punchline,plength,batch_first=True,enforce_sorted=False)

        optimizer.zero_grad()
        outputs = model(input_feature)
        loss = loss_fn(outputs,labels)

        train_running_loss += loss.clone().detach()
        train_running_acc += accuracy(outputs.clone().detach(),labels)
        loss.backward()
        optimizer.step()
    train_loss = train_running_loss/len(dataloader)
    train_acc = train_running_acc/len(dataloader)
    return train_loss,train_acc

def test(model,dataloader,full=False):
    model.eval()
    ground_truth = []
    all_predictions = []

    with torch.no_grad():
        for data in tqdm(dataloader,desc='Training...',total=len(dataloader),disable=False):
            punchline,sentence,labels,plength,slength = data['punchlines'].to('cuda'),data['sentences'].to('cuda'),data['labels'].to('cuda'),data['punchline_lengths'],data['sentence_lengths']
            if full:
                input_feature = (pack_padded_sequence(sentence,slength,batch_first=True,enforce_sorted=False))
            else:
                input_feature = (pack_padded_sequence(punchline,plength,batch_first=True,enforce_sorted=False))

            outputs = model(input_feature)
            predictions = (outputs>0.5).float()
            truths = labels.to('cpu')

            ground_truth.extend(truths.tolist())
            all_predictions.extend(predictions.tolist())

    return ground_truth,all_predictions

    

def main():

    parser = argparse.ArgumentParser(description="Train/evaluate TextLSTMClassifier")

    parser.add_argument('--filter', type=int, default = 0)
    parser.add_argument('--use_full',type=int, default=0)
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=4e-4)
    parser.add_argument("--epochs",         type=int,   default=10)
    parser.add_argument("--weight_decay",   type=float, default=1e-6)

    args = parser.parse_args()

    print("Training Text Model")

    print(args)

    with open('filtered.json','rb') as f:
        full_filtered_ids = json.load(f)
    with open('audio_filtered.json','rb') as f:
        audio_filtered_ids = json.load(f)

    text_pkls_dir = '../../../../../scratch1/jiaqilu/CSCI535/CSCI535-Project/dataset/urfunny2_text_feature_pkl/'

    for idx, id in enumerate(full_filtered_ids):
        full_filtered_ids[idx] = id.split('.')[0]
    for idx, id in enumerate(audio_filtered_ids):
        audio_filtered_ids[idx] = id.split('.')[0]

    # only doing this for testing to shorten the list
    #num_samples = int(len(full_filtered_ids)*0.1)
    #sampled_list = random.sample(full_filtered_ids,num_samples)
    #full_filtered_ids = sampled_list

    #print(len(full_filtered_ids))
        
    if args.filter == 0:
        filtered_ids = {}
    if args.filter == 1:
        filtered_ids = audio_filtered_ids
    if args.filter == 2:
        filtered_ids = full_filtered_ids

    with open('../dataset_extracted/humor_label_sdk.pkl','rb') as f:
        labels = pickle.load(f)

    #print(labels)

    punchline_data,sentence_data = load_pickle_data(text_pkls_dir,filtered_ids)
    #print(punchline_data)
    print(len(punchline_data),len(sentence_data))
    print(len(list(punchline_data.values())),len(list(sentence_data.values())))
    cleaned_labels = {}
    for key,value in labels.items():
        if len(filtered_ids) >0:
            if str(key) in filtered_ids:
                cleaned_labels[str(key)] = value
        else:
            if str(key) in list(punchline_data.keys()):
                cleaned_labels[str(key)] = value

    punchline_data = dict(sorted(punchline_data.items()))
    sentence_data = dict(sorted(sentence_data.items()))
    cleaned_labels = dict(sorted(cleaned_labels.items()))

    print(len(cleaned_labels),len(list(cleaned_labels.values())))

                          
    humor_dataset = TextDataset(list(punchline_data.values()),list(sentence_data.values()),list(cleaned_labels.values()))

    train_size = int(len(humor_dataset)*0.7)
    test_size = len(humor_dataset)-train_size

    generator = torch.Generator().manual_seed(42)
    batch_size = args.batch_size
    textmodel = TextLSTMClassifier(hidden_dim=1024,lstm_hidden=128,bidirectional=True).to('cuda')
    epochs = args.epochs
    lr = args.lr
    loss_function = nn.BCEWithLogitsLoss()
    weight_decay_rate = args.weight_decay
    optimizer = torch.optim.AdamW(textmodel.parameters(),weight_decay=weight_decay_rate,lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')

    train_dataset, test_dataset = torch.utils.data.random_split(humor_dataset,[train_size,test_size],generator=generator)

    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn = collate_fn_text)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,collate_fn = collate_fn_text)

    # start train loop
    train_loss = []
    train_acc = []
    full_or_not = False
    if args.use_full == 1:
        full_or_not = True
    for epoch in range(1,epochs+1):
        train_epoch_loss, train_epoch_acc = train(textmodel,train_dataloader,optimizer,loss_function,full=full_or_not)
        # not using validation and scheduler for now
        train_loss.append(train_epoch_loss.detach().cpu().numpy())
        train_acc.append(train_epoch_acc)
        print(f'Epoch: {epoch},\
              Train loss: {train_epoch_loss:.4f},\
              Train acc: {train_epoch_acc:.4f}')
    print('done training')

    test_labels, test_predictions = test(textmodel,test_dataloader,full=full_or_not)

    print(test_labels)
    print(test_predictions)
    correct = sum(1 for p, t in zip(test_predictions, test_labels) if p == t)
    test_acc = correct / len(test_labels)
    print(f"Test Acc: {test_acc*100:.2f}%")
    auc = roc_auc_score(test_labels, test_predictions)
    print(f"AUC: {auc*100:.2f}%")


if __name__=='__main__':
    main()