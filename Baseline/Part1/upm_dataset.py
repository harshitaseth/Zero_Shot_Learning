import os
import pickle
import json
import librosa
import numpy as np
import torch
import re
import torchaudio
import pickle as pkl
import torch.nn.functional as F
from omegaconf import OmegaConf
# from mm_music_captioning.utils.utils import load_conf, merge_conf
# from mm_music_captioning.utils.vocab import Vocabulary
# from mm_music_captioning.datasets.base_dataset import BaseDataset
from base_dataset import BaseDataset
# from mm_music_captioning.models.cnn_lstm_caption import CNNLSTMCaption
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer
import model_all as models


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
        root = "/efs_storage/Harshita/"
        self.w2v = pickle.load(open(os.path.join(root, 'google_wv.pkl'), "rb"))
        self.input_length =173
        self.skipped_caption = []
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        config = OmegaConf.load("model_config.yaml")
        train_data = pickle.load(open("./Pkls/train_features_20k_pretrained_encoders.pkl","rb"))
        val_data = pickle.load(open("./Pkls/val_features_20k_pretrained_encoders.pkl","rb"))[:2]
        self.audio_emb = train_data[:1][0]
        self.text_emb = train_data[1:2][0]
        self.audio_emb.extend(val_data[0])
        self.text_emb.extend(val_data[1])



        # self.model = models.MusCLAP(config.model_config).cuda()

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
        # import pdb;pdb.set_trace()
        tokens = tokenized_caption.split(" ")
        encoded_input = self.tokenizer(tokenized_caption, padding ="max_length", truncation=True,return_tensors="pt" )
        encoded_input["input_ids"] = encoded_input["input_ids"][:,:70]
        encoded_input["attention_mask"] = encoded_input["attention_mask"][:,:70]
        # emb = []
        # for token in tokens:
        #     try:
        #         token = re.sub('\W+','', token).lower()
        #         out = self.w2v.get_vector(token)
        #         emb.append(out)
        #     except:
        #         continue
        
        # try:
        #     emb = np.vstack(np.array(emb)).mean(axis = 0)
        # except:
        #     self.skipped_caption.append(tokenized_caption)
        #     pkl.dump(self.skipped_caption,open("skipped.pkl","wb"))
            # emb = np.zeros(300)+ 1e-6

        # token_idx = torch.ShortTensor([
        #     self.vocab.word2idx.get(
        #         token, self.vocab.UNK_INDEX)
        #     for token in tokenized_caption
        # ])
        # caption_length = len(token_idx)
        return encoded_input
        # return encoded_input["input_ids"], encoded_input['attention_mask']


    def get_audio(self, idx):
        
        audio = np.load(self.audio_paths[idx], mmap_mode="r").astype('float32')
        # audio = librosa.feature.melspectrogram(audio)
        # import pdb;pdb.set_trace()
        if len(audio) < 30*16000:
            audio = np.hstack((audio,[0]*(30*16000 - len(audio)))) # do random crop
        audio = torch.Tensor(audio[:30*16000])
        return audio

    def __getitem__(self, idx):
        """Returns one data pair (audio, caption)."""
        # audio = self.get_audio(idx)
        # token_idx = self.get_caption(idx)
        # import pdb;pdb.set_trace()
        audio = self.audio_emb[idx]
        token_idx = self.text_emb[idx]

       

        # import pdb;pdb.set_trace()
        # tag_emb = self.w2v[token_idx]
        # spec = self.load_spec(audio)
        return audio, token_idx

    def __len__(self):
        return 20000 #len(self.samples)

    @classmethod
    def config_path(cls):
        return "configs/datasets/upm.yaml"
