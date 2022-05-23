import os
import torch
import argparse
import model
import pickle
from torch.utils.data import DataLoader
from data_loader_eval import MyDataset
from torch import device, save
from losses import TripletLoss
from utils import *
import numpy as np
from torch import nn
from sklearn import metrics
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter()
instrument_list = ["acousticguitar", "electricpiano", "classicalguitar", "drummachine" "acousticbassguitar","electricguitar","doublebass","pipeorgan"]

def get_all_dataloader():
	
	train_loader = DataLoader(dataset=MyDataset(data_path, split='TRAIN', input_type=input_type, input_length=input_length, w2v_type=w2v_type), 
						  batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
	val_loader = DataLoader(dataset=MyDataset(data_path, split='VALID', input_type=input_type,
											input_length=input_length, num_chunk=num_chunk,
											w2v_type=w2v_type),
						  batch_size=batch_size//num_chunk, shuffle=False, drop_last=False, num_workers=num_workers)
	
	test_loader =[]
	return train_loader, val_loader, test_loader

def song_to_emb(spec, tags):
	instrument_label = torch.rand(41).reshape(1,-1)
	b, c, f, t = spec.size()
	if tags == True:
		out = model.spec_to_embedding(spec.view(-1, f, t))
	else:
		out,instrument_label = model.spec_to_embedding(spec.view(-1, f, t))
	out = out.view(b, c, -1)
	song_emb = out.mean(dim=1).detach().cpu()
	return song_emb, instrument_label

def tags_to_emb():
	tag_emb_out = []
	tags_out = []
	w2v = pickle.load(open(os.path.join(data_path, "google_emb.pkl"), 'rb'))
	for tag in all_tags:
		
		tag_emb = torch.from_numpy(w2v[tag.split("-")[-1]])
		tag_emb = model.word_to_embedding(tag_emb.unsqueeze(0)).detach().cpu()
		tag_emb_out.append(tag_emb)
		tags_out.append(tag)
	return tag_emb_out, tags_out

def tags_to_emb_music():
	tag_emb_out = []
	tags_out = []
	w2v = pickle.load(open(os.path.join(data_path, "google_emb_music.pkl"), 'rb'))
	for tag in w2v.keys():
		
		tag_emb = torch.from_numpy(w2v[tag.split("-")[-1]])
		tag_emb = model.word_to_embedding(tag_emb.unsqueeze(0)).detach().cpu()
		tag_emb_out.append(tag_emb)
		tag = "instrument---"+tag
		tags_out.append(tag)
	return tag_emb_out, tags_out

def get_similarity(tag_embs, song_embs):
	sim_scores = np.zeros((len(tag_embs), len(song_embs)))
	for i in range(len(tag_embs)):
		sim_scores[i] = np.array(nn.CosineSimilarity(dim=-1)(tag_embs[i], song_embs))
	return sim_scores

def get_list(tags,tags_out):
	gt = np.zeros(len(tags_out))
	for i in range(len(tags_out)):
		if tags_out[i] in tags:
			gt[i] = 1
	return gt

def get_list_music(tags,tags_out):
	gt = np.zeros(len(tags_out))
	for i in range(len(tags_out)):
		if tags_out[i] in tags:
			gt[i] = 1
	return gt


def val_model(val_loader,model):
	song_embs = []
	gt_tags = []
	p_ks = []
	gt_instruments = []
	pred_instruments  = []
	soft = nn.Softmax(dim=1)
	for i, batch in enumerate(val_loader):
		tags, _, spec,inst_gt,ids = batch
		if np.hstack(tags[0]) in instrument_list:
			print(np.hstack(tags[0]))
			continue
		spec = spec.unsqueeze(0)
		# if i > 5:
		# 	break
		song_emb,instrument_label = song_to_emb(spec, tags = False)

		pred = torch.argmax(soft(instrument_label))
		spec = spec.unsqueeze(0)
		song_embs.append(song_emb)
		gt_tags.append(tags)
		gt_instruments.append(inst_gt)
		pred_instruments.append(pred)
		break

	
	tag_embs,tags_out = tags_to_emb()
	song_embs = torch.cat(song_embs, dim=0)
	correct = 0
	incorrect = 0

	for i in range(len(song_embs)):
		embs = song_embs[i]
		inst_gt = gt_instruments[i]
		inst_pred = pred_instruments[i]
		gt_i = np.vstack(gt_tags[i]).reshape(-1)
		gt_all = get_list(gt_i, tags_out)
		gt = gt_all
		scores = np.array(nn.CosineSimilarity(dim=-1)(torch.cat(tag_embs), embs))
		sorted_ix = np.argsort(scores)[::-1][:10]
		pred = np.zeros(len(tags_out))
		pred[sorted_ix] = 1
		p_k = metrics.precision_score(gt, pred)
		p_ks.append(p_k)

		inst_label = reverse_instrument_label[inst_pred.numpy().tolist()]
	
		print(inst_label,np.vstack(inst_gt).reshape(-1))
		if inst_label in np.vstack(inst_gt).reshape(-1):
			correct += 1
		else:
			incorrect += 1

	print(p_ks)
	print("Instrument", correct/(correct+ incorrect))
	print("Precision", sum(p_ks)/len(p_ks))





if __name__ == '__main__':

	
	is_balanced = False
	w2v_type = 'google'
	margin = 0.4
	num_chunk = 1
	input_type = 'spec'
	n_epochs = 100
	input_length = 173
	is_subset = False
	data_path = "../Data/"

	
	checkpoints_path = "./checkpoints_instrument/5.pth"

	#### Parameters for training #####
	batch_size = 1
	num_workers = 0
	learning_rate = 0.01
	epochs = 200
	cuda_device = 0

	w2v = pickle.load(open(os.path.join(data_path, "google_emb.pkl"), 'rb'))
	all_tags = pickle.load(open(os.path.join(data_path,"All_tags.pkl"), 'rb'))
	### Dataset Loader ##############
	train_loader, val_loader, test_loader = get_all_dataloader()
	instrument_label = pickle.load(open(os.path.join(data_path,"instrument.pkl"), 'rb'))
	reverse_instrument_label = {}
	for key in instrument_label.keys():
		reverse_instrument_label[instrument_label[key]] = key


	##### Optimzers #################
	model = model.AudioModel()
	model.eval()
	model.load_state_dict(torch.load(checkpoints_path, map_location = torch.device("cpu")))

	
	########### Training ###############
	val_loss_out =  val_model(val_loader,model)
	