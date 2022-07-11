import os
from random import random
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import DataLoader, Subset

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne(audio_features, text_features):
    
    features = torch.cat((audio_features.float(), text_features.float()), dim=0).cpu().numpy()
    features = np.asarray(features, dtype='float64')
    import pdb;pdb.set_trace()
    y = torch.cat((torch.zeros(audio_features.shape[0]), torch.ones(text_features.shape[0])), dim=0).cpu().numpy()
    y_map = {0: "Audio", 1: "Text"}
    y = [y_map[i] for i in y]

    tsne_results = TSNE(n_components=2, init="random").fit_transform(features)
    # tsne_results = TSNE(n_components=2, learning_rate="auto", init="random").fit_transform(features.float())

    # Create the figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)

    # Create the scatter
    sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=y,
        # cmap=plt.cm.get_cmap("Paired"),
        alpha=0.3
        # s=0.5,
    )
    plt.legend(fontsize="large", title_fontsize="12")
    plt.savefig("./figures/train_features_20k_sgd_1e-1_10.png", dpi=300)

def plot_tsne_test(audio_features, text_features,audio_label):
    
    features = torch.cat((audio_features.float(), text_features.float()), dim=0).cpu().numpy()
    features = np.asarray(features, dtype='float64')
    import pdb;pdb.set_trace()
    y = torch.cat((torch.zeros(audio_features.shape[0]), torch.ones(text_features.shape[0])), dim=0).cpu().numpy()
    # y = audio_label
    y_map = {0: "Audio", 1: "Text"}
    y = [y_map[i] for i in y]

    tsne_results = TSNE(n_components=2, init="random").fit_transform(features)
    # tsne_results = TSNE(n_components=2, learning_rate="auto", init="random").fit_transform(features.float())

    # Create the figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)

    # Create the scatter
    sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=y,
        # cmap=plt.cm.get_cmap("Paired"),
        alpha=0.3
        # s=0.5,
    )
    plt.legend(fontsize="large", title_fontsize="12")
    plt.savefig("./figures/gtzan_Bert_pretrained_output_description_new_best.png", dpi=300)


if "__main__" == __name__:
    train_features = pkl.load(open("./Pkls/train_features_20k_sgd_1e-1_10.pkl","rb"))
    # audio = torch.stack(train_features[0]).detach().squeeze(1)
    # text = torch.stack(train_features[1]).detach().squeeze(1)
    # out = torch.stack(train_features[2]).detach().squeeze(1)
    
    # plot_tsne(out,text)

    import pdb;pdb.set_trace()
    
    # val_features = pkl.load(open("val_features.pkl","rb"))
    # audio = torch.stack(val_features[0]).detach().squeeze(1)
    # text = torch.stack(val_features[1]).detach().squeeze(1)
    # out = torch.stack(val_features[2]).detach().squeeze(1).cpu()
    # plot_tsne(out,text)

    test_features = pkl.load(open("../Evaluation/gtzan_Bert_pretrained_output_description_new_best.pkl","rb"))
    audio = torch.stack(test_features[0]).detach().squeeze(1)
    audio_label = np.array(test_features[1])
    text = torch.stack(test_features[2]).detach().squeeze(1).cpu()
    # text = torch.stack(train_features[1]).detach().squeeze(1)
    text_label = np.array(test_features[3])
    plot_tsne_test(audio, text,audio_label)