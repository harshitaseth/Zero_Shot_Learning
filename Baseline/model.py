import torch
from torch import nn
from modules import Conv_2d, Conv_emb



class AudioModel(nn.Module):
	def __init__(self):
		super(AudioModel, self).__init__()

		# CNN module for spectrograms
		self.spec_bn = nn.BatchNorm2d(1)
		self.layer1 = Conv_2d(1, 128, pooling=2)
		self.layer2 = Conv_2d(128, 128, pooling=2)
		self.layer3 = Conv_2d(128, 256, pooling=2)
		self.layer4 = Conv_2d(256, 256, pooling=2)
		self.layer5 = Conv_2d(256, 256, pooling=2)
		self.layer6 = Conv_2d(256, 512, pooling=2)
		self.layer7 = Conv_2d(512, 512, pooling=2)
		self.layer8 = Conv_emb(512, 256)
		# self.classifier = nn.Linear(256, 41)

		# FC module for word embedding
		self.fc1 = nn.Linear(300, 512)
		self.bn1 = nn.BatchNorm1d(512)
		self.fc2 = nn.Linear(512, 256)

		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.5)

	def spec_to_embedding(self, spec):
		
		
		
		out = spec.unsqueeze(1)
		out = self.spec_bn(out)

		# CNN
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.layer5(out)
		
		out = self.layer6(out)
		out = self.layer7(out)
		out = self.layer8(out)
		out = out.squeeze(2)
		out = nn.MaxPool1d(out.size(-1))(out)
		out = out.view(out.size(0), -1)
		# classification = self.classifier(out)
		
		return out

	def word_to_embedding(self, emb):
		
		out = self.fc1(emb)
		# out = self.bn1(out)
		out = self.relu(out)
		out = self.dropout(out)
		out = self.fc2(out)
		return out

	def forward(self, tag,spec):
		# import pdb;pdb.set_trace()
		tag_emb = self.word_to_embedding(tag)
		song_emb = self.spec_to_embedding(spec)
		return tag_emb,song_emb

class decoder(nn.Module):
	def __init__(self):
		super(decoder, self).__init__()
		self.spec_bn = nn.BatchNorm2d(1)
		self.fc1 = nn.Linear(256, 256)
		self.fc2 = nn.Linear(256, 512)
		self.fc3 = nn.Linear(512, 512)
		self.fc4 = nn.Linear(512, 256)
	def forward(self, emb):
		
		out = emb.unsqueeze(1)
		out = self.fc1(emb)
		out = self.fc2(out)
		out = self.fc3(out)
		out = self.fc4(out)
		
		return out

