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


class TripletLoss(nn.Module):

	def __init__(self, margin):
		super(TripletLoss, self).__init__()
		self.margin = margin
		self.relu = nn.ReLU()

	def forward(self, anchor, positive, negative, size_average=True):
		cosine_positive = nn.CosineSimilarity(dim=-1)(anchor, positive)
		cosine_negative = nn.CosineSimilarity(dim=-1)(anchor, negative)
		losses = self.relu(self.margin - cosine_positive + cosine_negative)
		return losses.mean()



def valid_loss(self, tag_emb, song_emb):
		sims = nn.CosineSimilarity(dim=-1)(tag_emb, song_emb)
		return 1 - sims.mean()