import os
import torch
import argparse
import model as models
from torch.utils.data import DataLoader
from data_loader import MyDataset
from torch import device, save
from losses import TripletLoss
from utils import *
from logger import Logger as logger
# from decoder_module import Decoder
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from upm_dataset import Upm
from torch.utils.data import Subset

writer = SummaryWriter()
def get_all_dataloader():
    
    train_loader = DataLoader(dataset=MyDataset(data_path, split='TRAIN', input_type=input_type, input_length=input_length, w2v_type=w2v_type), 
                          batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    val_loader = DataLoader(dataset=MyDataset(data_path, split='VALID', input_type=input_type,
                                            input_length=input_length, num_chunk=num_chunk,
                                            w2v_type=w2v_type),
                          batch_size=batch_size//num_chunk, shuffle=False, drop_last=False, num_workers=num_workers)
   
    return train_loader, val_loader

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def get_dataloader(dataset):
    train_loader = DataLoader(dataset["train"], 
                          batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    val_loader = DataLoader(dataset["val"], 
                          batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)

    return train_loader, val_loader

def train_model(loader, optimizer, model):
    running_loss = 0
    running_loss1 = 0
    running_loss2 = 0
    model.train()
  
    for i, batch in tqdm(enumerate(loader)):
        # import pdb;pdb.set_trace()
        optimizer.zero_grad()
        
        spec, tag  = batch
        tag = tag.cuda()#.unsqueeze(0)
        spec = spec.cuda()#.unsqueeze(0)
        
        tag_emb,song_emb = model(tag, spec)
        out = decoder(song_emb)
        # decoder(song_emb,tag_emb, lengths = 1)
        loss =  criterion(out, tag_emb)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        running_loss += batch_loss
        
       

    return running_loss/len(loader)



def val_model(val_loader,model):
    
    running_loss = 0
    running_loss1 = 0
    running_loss2 = 0
    for i, batch in enumerate(val_loader):

        spec, tag  = batch
        tag = tag.cuda()#.unsqueeze(0)
        spec = spec.cuda()#.unsqueeze(0)
        
        tag_emb,song_emb = model(tag, spec)
        out = decoder(song_emb)
        out = decoder(song_emb)
        loss =  criterion(out, tag_emb)
        batch_loss = loss.item()
        running_loss += batch_loss
        

    return running_loss/len(val_loader)


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
    batch_size = 1
    num_workers = 0
    learning_rate = 0.01
    epochs = 200
    cuda_device = 0
   

    ### Dataset Loader ##############
    # train_loader, val_loader = get_all_dataloader()
    dataset = Upm(None,"filtered")
    dataset_split =  train_val_dataset(dataset)
    train_loader, val_loader = get_dataloader(dataset_split)
    import pdb;pdb.set_trace()


    ##### Optimzers #################
    embed_size = 256
    hidden_size = 256
    vocab = ""
    num_layers = 3
    # decoder = Decoder(embed_size=embed_size, hidden_size=hidden_size, vocab_size=len(vocab), num_layers=num_layers, stateful=False)#.to(device)
    decoder = models.decoder().cuda()
    model = models.AudioModel().cuda()
   
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    triplet_loss = TripletLoss(margin = 0.1)
    criterion = nn.MSELoss()
    
    ########## Training ###############
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
        # if mlflow:
        metrics = {'train loss': train_loss, 'val loss': val_loss}
        logger.log_metrics("",metrics, epoch)
        # comment out since params exceed char limit in mlflow
        # self.logger.log_params()

        checkpoint = {
        'epoch': epoch + 1,
        'state_dict': decoder.state_dict(),
        'optimizer': optimizer.state_dict()
        }

        is_best = val_loss < min_loss
        if is_best:
            min_loss = val_loss
        # save checkpoint in appropriate path (new or best)
        # logsger.save_checkpoint("",state=checkpoint, is_best=is_best)
        if val_loss < min_loss:
            save(model.state_dict(), checkpoints_path + str(epoch) + ".pth")
            min_loss = val_loss
        pickle.dump(Final_train_loss, open("Final_train_loss.pkl","wb"))
        pickle.dump(Final_val_loss, open("Final_val_loss.pkl","wb"))