import os
import sys
import pickle
import tqdm
import numpy as np
import pandas as pd
import random
from torch.utils import data


class MyDataset(data.Dataset):
	def __init__(self, data_path, split='TRAIN', input_type='spec', input_length=None, num_chunk=16, w2v_type='google'):
		
		self.data_path = data_path
		self.split = split
		self.input_type = input_type
		self.input_length = input_length
		self.num_chunk = num_chunk
		# self.is_balanced = is_balanced
		self.w2v_type = w2v_type

	
		
		self.dataset_mapping = pickle.load(open(os.path.join(data_path,"data_mapping.pkl"), 'rb'))
		self.instrument_label = pickle.load(open(os.path.join(data_path,"instrument.pkl"), 'rb'))
		self.song_datapath = pickle.load(open(os.path.join(data_path,"songid_path.pkl"), 'rb'))
		self.all_tags = pickle.load(open(os.path.join(data_path,"All_tags.pkl"), 'rb'))


		# load ids
		if split == 'TRAIN':
			self.train_ids = pickle.load(open(os.path.join(data_path, 'sample_train_ids.pkl'),"rb"))
		elif split == 'VALID':
			self.eval_ids =pickle.load(open(os.path.join(data_path, 'val_ids.pkl'),"rb"))
		elif split == 'TEST':
			self.eval_ids = np.load(os.path.join(data_path,'test_ids.npy'))

		# load binaries
	

		# load tag embedding
		self.load_tag_emb()

		

	def load_tag_emb(self):
		# self.tags = np.load(os.path.join(self.data_path, self.prefix+'tags.npy'))
		self.w2v = pickle.load(open(os.path.join(self.data_path, "google_emb.pkl"), 'rb'))


	def load_spec(self, song_id):
		fn = self.song_datapath[song_id]
		
		spec_path = os.path.join(self.data_path,"spectrograms", fn, str(int(song_id.split("_")[-1]))+".npy" )
		length = self.input_length
		spec = np.load(spec_path)

		# for short spectrograms
		if spec.shape[1] < self.input_length:
			nspec = np.zeros((128, self.input_length))
			nspec[:, :spec.shape[1]] = spec
			spec = nspec

		# multiple chunks for validation loader
		if self.split == 'TRAIN' or self.split == 'VALID':
			time_ix = int(np.floor(np.random.random(1) * (spec.shape[1] - length)))
			spec = spec[:, time_ix:time_ix+length]
		elif self.split == 'TEST':
			hop = (spec.shape[1] - self.input_length) // self.num_chunk
			spec = np.array([spec[:, i*hop:i*hop+self.input_length] for i in range(self.num_chunk)])
		return spec



	def get_eval_item(self, index):
		eval_id = self.eval_ids[index]
		data = self.dataset_mapping[eval_id]
		tags =  data["Tags"]
		instrument = data["Instrument"]
		tag = random.choice(tags)
		neg_tags = sorted(set(self.all_tags) - set(tags))
		neg_tag = random.choice(neg_tags)
		instrument_label = []
		for inst in instrument:
			instrument_label.append(self.instrument_label[inst])

		# tag embedding
		# tag_emb = self.w2v[tag.split("-")[-1]]
		neg_tag_emb = self.w2v[neg_tag.split("-")[-1]]
		spec = self.load_spec(eval_id)
		
		return tags,neg_tag_emb, spec, instrument, eval_id
	def __getitem__(self, index):
		if self.split == 'TRAIN':
			
			tag_emb,neg_tag_emb, spec, instrument_label = self.get_train_item(index)
			return tag_emb.astype('float32'),neg_tag_emb.astype('float32'), spec.astype('float32'), instrument_label
		elif (self.split == 'VALID') or (self.split == 'TEST'):
			
			tags, neg_tag_emb, spec, instrument_label, eval_id= self.get_eval_item(index)
			# if len(tag_emb) <= 10:
			# 	arr = [""] * (10 -len(tag_emb))
			# 	tag_emb.extend(arr)
			# else:
			# 	tag_emb[:10]


			return tags,neg_tag_emb.astype('float32'), spec.astype('float32'), instrument_label,eval_id

	def __len__(self):
		if self.split == 'TRAIN':
			return 10000
		elif (self.split == 'VALID') or (self.split == 'TEST'):
			return len(self.eval_ids)


