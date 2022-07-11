import random
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from encoder.encoders import CNNEncoder
from encoder.decoder import decoder_baseline
from encoder.discriminator import Discriminator
from encoder.textual_heads import TextTransformer,Bert_TextTransformer

class MusCLAP(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.audio_backbone = CNNEncoder()
        self.textual_head = Bert_TextTransformer(config.text)
        self.fc = nn.Linear(768, 512)
        self.decoder = decoder_baseline()
        audio_config = config.audio
        text_config = config.text
        projection_dim = config.projection_dim
        audio_dim = audio_config.hidden_size
        text_dim = text_config.hidden_size

        self.audio_projection = nn.Linear(audio_dim, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_dim, projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def encode_audio(self, audio):
        audio_features = self.audio_backbone(audio, audio.shape)
        # audio_features = self.audio_projection(audio_features)
        return audio_features

    def encode_text(self, text, text_mask):
        # x.shape = [batch_size, n_ctx, transformer.width]
        if isinstance(self.textual_head, Bert_TextTransformer): # TextTransformer --> Bert_TextTransformer
            text_features = self.textual_head(text, text_mask)
            # TODO check pooled output is correct (here taken as EOS/EOT token)
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # pooled_outout = text_features[
                # torch.arange(text_features.shape[0]), text.argmax(dim=-1)
            # ]
        elif isinstance(self.textual_head, CLIPTextModel):
            outputs = self.textual_head(text, text_mask)
            pooled_outout = outputs.pooler_output

        # text_features = self.text_projection(pooled_outout)
        return text_features

    def forward(
        self,
        audio,
        text,
        original_audio=None,
        sentence_sim=None,
        text_mask=None,
        return_loss=False,
    ):
        if return_loss:
            audio_ssl_loss = (
                self.audio_ssl(audio, original_audio) if self.do_audio_ssl else 0
            )

        # import pdb;pdb.set_trace()
        audio_features = self.encode_audio(audio)[0]
        # text = self.fc(text).long()
        text_features = self.encode_text(text, text_mask)
        text_features = self.fc(text_features)
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        emb_out = self.decoder(audio_features)
        

        return emb_out, text_features
class Model_gen(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        
        # self.fc = nn.Linear(768, 512)
        self.decoder = decoder_baseline()
        audio_config = config.audio
        
        
    def forward(self, audio_features):
        
        emb_out = self.decoder(audio_features)
        return emb_out


class Model_dis(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        
        # self.fc = nn.Linear(768, 512)
        self.decoder = Discriminator()

        
        
    def forward(self, audio_features):
        
        emb_out = self.decoder(audio_features)
        return emb_out
