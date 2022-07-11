import os
import random
import torch
import time
import pickle
import tqdm
import numpy as np
from sklearn import metrics
from torch import nn
from torch.nn import functional as F

data_path = ""
def load_eval_data(self, data_path, w2v_type, mode):
	# get w2v word embedding
	emb_dict = pickle.load(open(os.path.join(data_path, self.prefix+'%s_emb.pkl'%w2v_type), 'rb'))
	self.word_emb = torch.tensor([emb_dict[key] for key in emb_dict.keys()]).cuda()
	# get valid data
	if mode == 'TRAIN':
		self.eval_ids = np.load(os.path.join(data_path, self.prefix+'valid_ids.npy'))
	elif mode == 'TEST':
		self.eval_ids = np.load(os.path.join(data_path, self.prefix+'test_ids.npy'))
	self.tags = np.load(os.path.join(data_path, self.prefix+'tags.npy'))

	# preprocess
	binaries = np.load(os.path.join(data_path, self.prefix+'binaries.npy'))
	indice = [int(line.split('//')[0]) for line in self.eval_ids]
	self.ground_truth = binaries[indice]
	if self.input_type != 'spec':
		ix_to_cf = np.load(os.path.join(data_path, self.prefix+'ix_to_cf.npy'))
		self.ix_to_cf = ix_to_cf[indice]


def song_to_emb(self, spec, cf):
	b, c, f, t = spec.size()
	out = self.model.spec_to_embedding(spec.view(-1, f, t))
	out = out.view(b, c, -1)
	song_emb = out.mean(dim=1).detach().cpu()
	return song_emb

def tags_to_emb(model):
	w2v = pickle.load(open(os.path.join(data_path, "google_emb.pkl"), 'rb'))
	tag_emb = model.word_to_embedding(w2v).detach().cpu()
	return tag_emb

def get_similarity(tag_embs, song_embs):
	sim_scores = np.zeros((len(tag_embs), len(song_embs)))
	for i in range(len(tag_embs)):
		sim_scores[i] = np.array(nn.CosineSimilarity(dim=-1)(tag_embs[i], song_embs))
	return sim_scores

def get_scores(tag_embs, song_embs,gt_tags):
	# get similarity score (tag x song)
    sim_scores = get_similarity(tag_embs, song_embs)

    # get metrics
    k = 10
    p_ks = get_precision(sim_scores, k=k)
    roc_aucs = get_roc_auc(sim_scores)
    aps = get_ap(sim_scores)

    # print
    print('precision @%d: %.4f' % (k, np.mean(p_ks)))
    print('roc_auc: %.4f' % np.mean(roc_aucs))
    print('map: %.4f' % np.mean(aps))
    for i, tag in enumerate(self.tags):
        print('%s: %.1f, %.4f, %.4f' % (tag, p_ks[i], roc_aucs[i], aps[i]))
    return torch.tensor(np.mean(p_ks)), torch.tensor(np.mean(roc_aucs)), torch.tensor(np.mean(aps))
def get_precision(sim_scores, k=10):
	p_ks = []
	for i in range(len(sim_scores)):
		sorted_ix = np.argsort(sim_scores[i])[::-1][:k]
		gt = self.ground_truth.T[i][sorted_ix]
		p_k = metrics.precision_score(gt, np.ones(k))
		p_ks.append(p_k)
	return p_ks

def get_roc_auc(self, sim_scores):
		return metrics.roc_auc_score(self.ground_truth, sim_scores.T, average=None)


def get_ap(self, sim_scores):
		return metrics.average_precision_score(self.ground_truth, sim_scores.T, average=None)


def triplet_sampling(self, tag_emb, song_emb, tag_binary, song_binary):
		num_batch = len(tag_emb)
		if self.is_weighted:
			# get distance weights
			tag_norm = tag_emb / tag_emb.norm(dim=1)[:, None]
			song_norm = song_emb / song_emb.norm(dim=1)[:, None]
			dot_sim = torch.matmul(tag_norm, song_norm.T)
			weights = (dot_sim + 1) / 2

			# masking
			mask = 1 - torch.matmul(tag_binary, song_binary.T)
			masked_weights = weights * mask

			# sampling
			index_array = torch.arange(num_batch)
			negative_ix = [random.choices(index_array, weights=masked_weights[i], k=1)[0].item() for i in range(num_batch)]
		else:
			# masking
			mask = 1 - torch.matmul(tag_binary, song_binary.T)

			# sampling
			index_array = torch.arange(num_batch)
			negative_ix = [random.choices(index_array, weights=mask[i], k=1)[0].item() for i in range(num_batch)] 
		negative_emb = song_emb[negative_ix]
		return tag_emb, song_emb, negative_emb