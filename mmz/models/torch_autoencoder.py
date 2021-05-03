import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from mmz.models import BaseTrainer

#from acgan import torch_gan

import torch.utils.data

import numpy as np
import attr
from tqdm.auto import tqdm

class AutoEncoder(nn.Module):
    def __init__(self, input_size=784, hidden_size=400, z_size=10,
                 hidden_activation=nn.LeakyReLU, output_activation=nn.Sigmoid):
        super(AutoEncoder, self).__init__()

        self.input_size = input_size

        #### Encoding Params
        self.encoder = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(input_size, hidden_size),
            #nn.BatchNorm1d(hidden_size),
            hidden_activation(),
            nn.Linear(hidden_size, z_size),
            #nn.BatchNorm1d(hidden_size),
            hidden_activation(),
        )
        #self.z_dim = nn.Linear(hidden_size, z_size)

        #### Decoding Params
        self.decoder = nn.Sequential(
            nn.Linear(z_size, hidden_size),
            #nn.BatchNorm1d(hidden_size),
            hidden_activation(),
            nn.Linear(hidden_size, input_size),
            #nn.BatchNorm1d(hidden_size),
            output_activation()
        )

    def forward(self, x):
        #z = self.encoder(x.view(-1))
        z = self.encoder(x)
        decoded = self.decoder(z)
        return decoded, z



@attr.attrs
class AutoEncoderTrainer(BaseTrainer):
#    model = attr.ib()
#
#    data_gen = attr.ib()
#    learning_rate = attr.ib(0.0002)
#    beta1 = attr.ib(0.5)
#    n_samples = attr.ib(None)
#    device = attr.ib(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    criterion = attr.ib(torch.nn.MSELoss())
#
#    epochs_trained = attr.ib(0, init=False)
    def train_inner_step(self, epoch_i, data_batch):
        enc_model = self.model_map['enc']
        dec_model = self.model_map['dec']
        enc_opt, dec_opt = self.opt_map['enc'], self.opt_map['dec']

        enc_opt.zero_grad(), dec_opt.zero_grad()

        x = data_batch[0].to(self.device)
        embed_batch = enc_model(x)
        recon_batch = dec_model(embed_batch)
        #recon_batch, embed_batch = model(data)
        loss = self.criterion(recon_batch, x)

        loss.backward()
        #train_loss += loss.item()
        enc_opt.step()
        dec_opt.step()

        return dict(loss=loss.item())


#    def train_old(self, n_epochs):
#
#            self.model.to(self.device)
#            self.model.train()
#            train_loss = 0
#            self.optim = torch.optim.Adam(self.model.parameters(),
#                                          lr=self.learning_rate,
#                                          betas=(self.beta1, 0.999))
#            try:
#                n_batches_per_epoch = len(self.data_gen)
#            except:
#                n_batches_per_epoch = self.n_samples
#
#            losses = list()
#            with tqdm(total=n_epochs,
#                      desc='Training epoch') as epoch_pbar:
#                for epoch in range(self.epochs_trained, self.epochs_trained + n_epochs):
#
#                    with tqdm(total=n_batches_per_epoch, desc='-loss-') as batch_pbar:
#                        for batch_idx, (data, _) in enumerate(self.data_gen):
#                            data = data.to(self.device)
#                            self.optim.zero_grad()
#
#                            recon_batch, embed_batch = self.model(data)
#                            loss = self.criterion(data, recon_batch)
#
#                            loss.backward()
#                            train_loss += loss.item()
#                            self.optim.step()
#
#                            ### Save Losses
#                            losses.append(loss.item())
#                            batch_pbar.set_description("Loss: %.6f"
#                                                       % (np.mean(losses[-20:])))
#                            batch_pbar.update(1)
#
#                    epoch_pbar.update(1)
#            self.model.eval()
#            return losses


