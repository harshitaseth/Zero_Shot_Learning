import os
import pickle
import json
import librosa
import numpy as np
import torch
import re
import torchaudio
import torch.nn.functional as F

# from mm_music_captioning.utils.utils import load_conf, merge_conf
# from mm_music_captioning.utils.vocab import Vocabulary
# from mm_music_captioning.datasets.base_dataset import BaseDataset
from base_dataset import BaseDataset
# from mm_music_captioning.models.cnn_lstm_caption import CNNLSTMCaption


class Upm(BaseDataset):
    def __init__(self, config, dataset_type="train"):
        """ Constructs a Upm dataset, inheriting from BaseDataset.
        Args:
        - config: 
        - vocab: vocabulary object 
        - dataset_type: "train", "test" or "val"
        """
        super().__init__(
            config, dataset_type, dataset_name="upm")

        self.dataset_json = os.path.join(
            self._data_dir, "dataset_{}.json".format(self._dataset_type))
        self._load()
        root = "/efs_storage/Harshita/Zero_Shot_Learning"
        self.w2v = pickle.load(open(os.path.join(root, 'google_wv.pkl'), "rb"))
        self.input_length =173
    def _load(self):
        with open(self.dataset_json) as f:
            self.samples = json.load(f)
            # self._build_vocab()

            # self.captions = [[self.vocab.SOS_TOKEN] + i["caption"]['tokens'] +
            #                  [self.vocab.EOS_TOKEN] for i in self.samples]
            
            self.captions = [[i["caption"]for i in self.samples]][0]

            self.audio_dir = "/data/maml-ilaria_dataset"
            self.audio_paths = [os.path.join(self.audio_dir, i['audio_path'].split(
                "/")[-1].replace(".mp3", ".npy")) for i in self.samples]

    def _build_vocab(self):
        """ Build vocab based on captions in the training set"""
        if self._dataset_type == "train":
            self.vocab = Vocabulary(
                [i["caption"]['tokens'] for i in self.samples])
        else:
            training_set = os.path.join(
                self._data_dir, "dataset_train.json")
            with open(training_set) as f:
                samples = json.load(f)
                training_captions = [
                    i["caption"]['tokens'] for i in samples]
            self.vocab = Vocabulary(
                training_captions)

    def get_caption(self, idx):
        """Get caption and convert list of strings to tensor of word indices"""
        # TODO move this somewhere else?
        tokenized_caption = self.captions[idx]
        
        tokens = tokenized_caption.split(" ")
        emb = []
        for token in tokens:
            try:
                token = re.sub('\W+','', token).lower()
                out = self.w2v.get_vector(token)
                emb.append(out)
            except:
                continue
        emb = np.vstack(np.array(emb)).mean(axis = 0)
        # token_idx = torch.ShortTensor([
        #     self.vocab.word2idx.get(
        #         token, self.vocab.UNK_INDEX)
        #     for token in tokenized_caption
        # ])
        # caption_length = len(token_idx)
        return emb

    # def load_spec(self, song_id):
	# 	fn = self.song_datapath[song_id]
		
	# 	spec_path = os.path.join(self.data_path,"spectrograms", fn, str(int(song_id.split("_")[-1]))+".npy" )
	# 	length = self.input_length
	# 	spec = np.load(spec_path)

	# 	# for short spectrograms
	# 	if spec.shape[1] < self.input_length:
	# 		nspec = np.zeros((128, self.input_length))
	# 		nspec[:, :spec.shape[1]] = spec
	# 		spec = nspec

	# 	# multiple chunks for validation loader
	# 	if self.split == 'TRAIN' or self.split == 'VALID':
	# 		time_ix = int(np.floor(np.random.random(1) * (spec.shape[1] - length)))
	# 		spec = spec[:, time_ix:time_ix+length]
	# 	elif self.split == 'TEST':
	# 		hop = (spec.shape[1] - self.input_length) // self.num_chunk
	# 		spec = np.array([spec[:, i*hop:i*hop+self.input_length] for i in range(self.num_chunk)])
	# 	return spec

    def get_audio(self, idx):
        
        audio = np.load(self.audio_paths[idx]).astype('float32')
        spec = librosa.feature.melspectrogram(audio)

        audio = torch.Tensor(spec)
        return audio

    def __getitem__(self, idx):
        """Returns one data pair (audio, caption)."""
        audio = self.get_audio(idx)
        token_idx = self.get_caption(idx)
        # tag_emb = self.w2v[token_idx]
        # spec = self.load_spec(audio)
        return audio, token_idx

    def __len__(self):
        return 100 #len(self.samples)

    @classmethod
    def config_path(cls):
        return "configs/datasets/upm.yaml"
