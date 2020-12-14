import torch
#from torch import data as tdata
from torch.utils import data as tdata

class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


import attr
from tqdm.auto import tqdm
import numpy as np

@attr.attrs
class BaseTrainer():
    model_map = attr.ib()
    opt_map = attr.ib()


    #gen_model = attr.ib()
    #disc_model = attr.ib()
    data_gen = attr.ib()
    in_channel_size = attr.ib(100)
    in_channel_dim = attr.ib(3)
    n_samples = attr.ib(None)
    learning_rate = attr.ib(0.0003)
    beta1 = attr.ib(0.5)
    criterion = attr.ib(torch.nn.BCELoss())
    disc_optim = attr.ib(None)
    gen_optim = attr.ib(None)
    device = attr.ib(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    epochs_trained = attr.ib(0, init=False)

    def __attrs_post_init__(self):
        self.model_map = {k: v.to(self.device) for k, v in self.model_map.items()}
        #self.opt_map = dict()
        for k, m in self.model_map.items():
            m.apply(self.weights_init)

            if k not in self.opt_map:
                self.opt_map[k] = torch.optim.Adam(m.parameters(),
                                                   lr=self.learning_rate,
                                                   betas=(self.beta1, 0.999))



    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if 'Conv' in classname or 'Linear' in classname:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif 'BatchNorm' in classname:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)

    def sample_z(self, batch_size):
        if self.in_channel_dim > 1:
            single_dims = [1] * (self.in_channel_dim - 1)
            # noise = torch.randn(batch_size, self.in_channel_size,
            #                    *single_dims,
            #                    device=self.device)
            size = [batch_size, self.in_channel_size] + single_dims
            noise = torch.normal(0, 1, size=tuple(size), device=self.device)
        else:
            # noise = torch.randn(batch_size, self.in_channel_size,
            #                    device=self.device)
            size = [batch_size, self.in_channel_size]
            noise = torch.normal(0, 1, size=tuple(size), device=self.device)

        return noise

    def train_inner_step(self, epoch_i, data_batch):
        real_label = 1
        fake_label = 0

        disc_model = self.model_map['disc']
        gen_model = self.model_map['gen']
        disc_optim = self.opt_map['disc']
        gen_optim = self.opt_map['gen']

        disc_model.zero_grad()

        # Format batch
        real_cpu = data_batch[0].to(self.device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=self.device,
                           dtype=torch.float)
        # Forward pass real batch through D
        output = disc_model(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = self.criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        # if self.in_channel_dim > 1:
        #    single_dims = [1] * (self.in_channel_dim - 1)
        #    noise = torch.randn(b_size, nz, *single_dims,
        #                        device=self.device)
        #    #noise = torch.normal()
        # else:
        #    noise = torch.randn(b_size, nz, device=self.device)
        noise = self.sample_z(b_size)

        # Generate fake image batch with G
        fake = gen_model(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = disc_model(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = self.criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        disc_optim.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        gen_model.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = disc_model(fake).view(-1)
        # Calculate G's loss based on this output
        errG = self.criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        # D_G_z2 = output.mean().item()
        # Update G
        gen_optim.step()

        return dict(gen_l=errG.item(), disc_l=errD.item())


    def train(self, n_epochs, epoch_callbacks=None, batch_callbacks=None,
              batch_cb_delta=3):

        epoch_callbacks = dict() if epoch_callbacks is None else epoch_callbacks
        batch_callbacks = dict() if batch_callbacks is None else batch_callbacks

        #self.epoch_losses = list()
        self.train_batch_results = list()
        epoch_cb_history = [{k: cb(self, 0) for k, cb in epoch_callbacks.items()}]
        batch_cb_history = [{k: cb(self, 0) for k, cb in batch_callbacks.items()}]

        if self.n_samples is None:
            try:
                self.n_samples = len(self.data_gen)
            except:
                print("Unable to determine generator lengthe using __len__")

        epoch_results = dict(epoch=list(), batch=list())
        with tqdm(total=n_epochs,
                  desc='Training epoch') as epoch_pbar:
            for epoch in range(self.epochs_trained, self.epochs_trained + n_epochs):
                with tqdm(total=self.n_samples, desc='-loss-') as batch_pbar:
                    for i, data in enumerate(self.data_gen):
                        update_d = self.train_inner_step(epoch, data)

                        epoch_results['epoch'].append(epoch)
                        epoch_results['batch'] = i

                        prog_msgs = list()
                        for k, v in update_d.items():
                            if k not in epoch_results:
                                epoch_results[k] = [v]
                            else:
                                epoch_results[k].append(v)


                            v_l = np.round(np.mean(epoch_results[k][-20:]), 4)
                            prog_msgs.append(f"{k}: {v_l}")


                        msg = " || ".join(prog_msgs)
                        # Save Losses for plotting later
                        #G_losses.append(errG.item())
                        #D_losses.append(errD.item())
                        #batch_pbar.set_description("Gen-L: %.3f || Disc-L:%.3f" % (np.mean(G_losses[-20:]),
                        #                                                           np.mean(D_losses[-20:])))
                        batch_pbar.set_description(msg)
                        batch_pbar.update(1)
                        if not i % batch_cb_delta:
                            batch_cb_history.append({k: cb(self, epoch) for k, cb in batch_callbacks.items()})

                #self.epoch_losses.append(dict(gen_losses=G_losses, disc_losses=D_losses))
                #self.epoch_losses.append(epoch_results)
                #self.train_batch_results.append(epoch_results)
                self.epochs_trained += 1
                epoch_cb_history.append({k: cb(self, epoch) for k, cb in epoch_callbacks.items()})
                epoch_pbar.update(1)
        #return self.train_batch_results
        return epoch_results

    @staticmethod
    def grid_display(data, figsize=(10, 10),
                     pad_value=0, title='',
                     xlabel='', ylabel='',
                     nrow=8):
        from torchvision import utils as vutils
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.imshow(np.transpose(vutils.make_grid(data,
                                                padding=2,
                                                nrow=nrow,
                                                normalize=True,
                                                pad_value=pad_value).detach().cpu(), (1, 2, 0)))

        return fig

    def generate(self, noise=None, batch_size=1, gen_model_k='gen'):
        gen_model = self.model_map[gen_model_k]
        if noise is None:
            noise = torch.randn(batch_size, self.in_channel_size,
                                1, 1, device=self.device)
            fake_batch = gen_model(noise).to('cpu')
        else:
            fake_batch = gen_model(noise).to('cpu')

        return fake_batch

    def display_batch(self, batch_size=16, noise=None, batch_data=None, figsize=(10, 10), title='Generated Data',
                      pad_value=0):
        if batch_data is not None:
            fake_batch = batch_data
        else:
            # Generate fake image batch with G
            fake_batch = self.generate(noise=noise, batch_size=batch_size)

        return self.grid_display(fake_batch, figsize=figsize, title=title, pad_value=pad_value)