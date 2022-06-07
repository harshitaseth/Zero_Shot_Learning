import os
import torch
import argparse
import model
from torch.utils.data import DataLoader
from data_loader import MyDataset
from torch import device, save
from losses import TripletLoss
from utils import *
from decoder_module import Decoder
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
def get_all_dataloader():
    
    train_loader = DataLoader(dataset=MyDataset(data_path, split='TRAIN', input_type=input_type, input_length=input_length, w2v_type=w2v_type), 
                          batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    val_loader = DataLoader(dataset=MyDataset(data_path, split='VALID', input_type=input_type,
                                            input_length=input_length, num_chunk=num_chunk,
                                            w2v_type=w2v_type),
                          batch_size=batch_size//num_chunk, shuffle=False, drop_last=False, num_workers=num_workers)
   
    return train_loader, val_loader

def train_model(loader, optimizer, model):
    running_loss = 0
    running_loss1 = 0
    running_loss2 = 0
    model.train()
  
    for i, batch in tqdm(enumerate(loader)):
        optimizer.zero_grad()
        
        tag, neg, spec, instrument_label = batch
        tag = tag.cuda()
        neg = neg.cuda()
        spec = spec.cuda()
        instrument_label = instrument_label.cuda()
        tag_emb,neg_emb, song_emb,classification = model(tag,neg, spec)
        out = decoder(song_emb)
        # decoder(song_emb,tag_emb, lengths = 1)
        loss1 = triplet_loss(song_emb, tag_emb, neg_emb)
        loss2 =  criterion(classification, instrument_label)
        loss = loss1+loss2
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        running_loss += batch_loss
        
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
        # print(running_loss)

    return running_loss/len(loader), running_loss1/len(loader), running_loss2/len(loader)




def val_model(val_loader,model):
    
    running_loss = 0
    running_loss1 = 0
    running_loss2 = 0
    for i, batch in enumerate(val_loader):

        tag, neg, spec, instrument_label = batch
        tag = tag.cuda()
        neg = neg.cuda()
        spec = spec.cuda()
        instrument_label = instrument_label.cuda()
        tag_emb,neg_emb, song_emb,classification = model(tag,neg, spec)
        loss1 = triplet_loss(song_emb, tag_emb, neg_emb)
        loss2 =  criterion(classification, instrument_label)
        loss = loss1+loss2
        batch_loss = loss.item()
        running_loss += batch_loss
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
        

    return running_loss/len(val_loader), running_loss1/len(val_loader), running_loss2/len(val_loader)


if __name__ == '__main__':
    
    
    is_balanced = False
    w2v_type = 'google'
    margin = 0.4
    num_chunk = 1
    input_type = 'spec'
    n_epochs = 100
    input_length = 173
    is_subset = False
    data_path = "/data/MTG_baseline_dataset/Data/"
   
    checkpoints_path = "./checkpoints/iter1/"

    #### Parameters for training #####
    batch_size = 2
    num_workers = 0
    learning_rate = 0.01
    epochs = 200
    cuda_device = 0
   

    ### Dataset Loader ##############
    train_loader, val_loader = get_all_dataloader()



    ##### Optimzers #################
    embed_size = 256
    hidden_size = 256
    vocab = ""
    num_layers = 3
    decoder = Decoder(embed_size=embed_size, hidden_size=hidden_size, vocab_size=len(vocab), num_layers=num_layers, stateful=False)#.to(device)
    model = model.AudioModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    triplet_loss = TripletLoss(margin = 0.1)
    criterion = nn.CrossEntropyLoss()
    
    ########### Training ###############
    Final_train_loss = []
    Final_val_loss = []
    min_loss = 100
    for epoch in range(epochs):
        train_loss = train_model(train_loader, optimizer, model)
        val_loss_out =  val_model(val_loader,model)
        val_loss = val_loss_out
        writer.add_scalar("Loss", train_loss, epoch)
        Final_train_loss.append(train_loss)
        Final_val_loss.append(val_loss)
        print("Epoch, Train, Val", epoch, train_loss, val_loss)
        # import pdb;pdb.set_trace()
        if val_loss[0] < min_loss:
            save(model.state_dict(), checkpoints_path + str(epoch) + ".pth")
            min_loss = val_loss[0]
    pickle.dump(Final_train_loss, open("Final_train_loss.pkl","wb"))
    pickle.dump(Final_val_loss, open("Final_val_loss.pkl","wb"))