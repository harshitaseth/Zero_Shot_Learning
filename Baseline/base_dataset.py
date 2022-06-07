from torch.utils.data.dataset import Dataset
import torch


class BaseDataset(Dataset):
    def __init__(self, config, dataset_type, dataset_name):
        """Base class for a dataset.

        Args:
        - config: 
        - dataset_type: "train", "test" or "val"
        - dataset_name: string of the dataset name (only "westone" and "upm" supported)
        """
        super().__init__()
        if config is None:
            config = {}
        self.config = config
        self._dataset_name = dataset_name
        self._dataset_type = dataset_type
        self._data_dir = "/efs_storage/"

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        __getitem__ of a torch dataset.
        Args:
            idx (int): Index of the sample to be loaded.
        """

        raise NotImplementedError


class CustomCollator():
    def __init__(self, config):
        self.precomp_features = config.dataset_config.audio.precomp_features
        self.feature_dim = config.model_config.audio_feature_dim

    def __call__(self, data):
        """ Custom collate function for data loader to create mini-batch tensors of the same shape. 

        Args:
        data: list of tuple (audio, caption). 
            - audio: torch tensor of shape (?); variable length.
            - caption: torch tensor of shape (?); variable length.
        Returns:
            padded_audio: torch tensor of shape (batch_size, padded_audio_length).
            padded_captions: torch tensor of shape (batch_size, padded_sen_length).
        """

        # Sort a data list by audio length (descending order).
        data.sort(key=lambda x: len(x[1]), reverse=True)
        audio_tracks, captions = zip(*data)

        audio_lengths = [len(audio) for audio in audio_tracks]

        if self.precomp_features:
            # precomputed musicnn features have shape (batch_size, length/60, 753)
            max_audio_length = 365
            padded_audio = torch.zeros(len(audio_tracks), max_audio_length,
                                       753).float()
        else:
            # raw audio has shape (batch_size, length)
            # TODO test for eval
            max_audio_length = 360 * 16000
            # max_audio_length = max(audio_lengths)
            padded_audio = torch.zeros(
                len(audio_tracks), max_audio_length).float()

        # Pad and merge captions and audio (from tuple of 1D tensor to 2D tensor).
        cap_lengths = [len(cap) for cap in captions]
        padded_captions = torch.zeros(len(captions), max(cap_lengths)).long()
        for i, cap in enumerate(captions):
            caption_end = cap_lengths[i]
            padded_captions[i, :caption_end] = cap

            audio_end = audio_lengths[i]
            padded_audio[i, :audio_end] = audio_tracks[i]

        audio_lengths = torch.Tensor(audio_lengths).long()

        return padded_audio, audio_lengths, padded_captions, cap_lengths
