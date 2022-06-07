import os
import torch
import time
import mlflow
import mlflow.pytorch
from datetime import datetime
from omegaconf import OmegaConf

# from music_and_language.utils.utils import save_json


def get_epoch_time():
    return int((datetime.now() - datetime(1970, 1, 1)).total_seconds())

def save_json(output_path, content):
    with open(output_path, 'w') as outfile:
        json.dump(content, outfile)

class Logger:
    def __init__(self, config):
        self.config = config

        if "env" in self.config.keys():
            self.init_pretrain_logger()
        else:
            self.init_downstream_logger()
        self.init_training_log()

    def init_pretrain_logger(self):
        if self.config.env.experiment_id is None:
            self.experiment_id = self.get_timestamp()
            OmegaConf.update(self.config, "env.experiment_id",
                             self.experiment_id)
        else:
            self.experiment_id = self.config.env.experiment_id
        self.experiment_dir = os.path.join(self.config.env.experiments_dir,
                                           self.experiment_id)
        self.vocab_path = os.path.join(self.experiment_dir, 'vocab.json')

        self.checkpoint_path = os.path.join(
            self.experiment_dir, 'checkpoint.pth.tar')

    def init_downstream_logger(self):
        if self.config.downstream_experiment_id is None:
            self.experiment_id = self.get_timestamp()
            OmegaConf.update(self.config, "downstream_experiment_id", self.experiment_id)
        else:
            self.experiment_id = self.config.downstream_experiment_id

        self.experiment_dir = os.path.join(
            self.config.downstream_dir, self.experiment_id)
        self.checkpoint_path = os.path.join(
            self.experiment_dir, 'checkpoint.pth.tar')

    def init_training_log(self):
        self.log_filename = os.path.join(self.experiment_dir, "train_log.tsv")
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
            log_file = open(self.log_filename, 'a')
            log_file.write(
                'Epoch\ttrain_loss\tval_loss\tmetric\tepoch_time\tlearing_rate\ttime_stamp\n'
            )
            log_file.close()

    def save_config(self):
        # Save json with experiment settings
        config_path = os.path.join(self.experiment_dir, 'config.yaml')
        if not os.path.exists(config_path):
            OmegaConf.save(self.config, config_path)

    def get_timestamp(self):
        return str(time.strftime('%Y-%m-%d-%H_%M_%S', time.gmtime()))

    def write(self, text):
        print(text)

    def save_vocab(self, token_freq):
        save_json(self.vocab_path, token_freq)

    def update_training_log(self, epoch, train_loss, val_loss, epoch_time,
                            learning_rate, metric=0):
        time_stamp = str(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))
        self.write(
            'Epoch %d, train loss %g, val loss %g, metric %g, epoch-time %gs, lr %g, time-stamp %s'
            %
            (epoch, train_loss, val_loss, metric, epoch_time, learning_rate, time_stamp))

        log_file = open(self.log_filename, 'a')
        log_file.write('%d\t%g\t%g\t%g\t%gs\t%g\t%s\n' %
                       (epoch, train_loss, val_loss, metric, epoch_time, learning_rate,
                        time_stamp))
        log_file.close()

    def log_metrics(self, metrics, epoch):
        mlflow.log_metrics(metrics, step=epoch)

    def log_params(self, params):
        mlflow.log_params(params)

    def save_checkpoint(self, state, is_best=False):
        torch.save(state, self.checkpoint_path)
        if is_best:
            self.write("Saving best model so far")
            best_model_path = os.path.join(self.experiment_dir,
                                           'best_model.pth.tar')
            torch.save(state, best_model_path)
