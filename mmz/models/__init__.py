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
    in_trailing_dims = attr.ib((1, 1))
    n_samples = attr.ib(None)
    learning_rate = attr.ib(0.0003)
    beta1 = attr.ib(0.5)
    criterion = attr.ib(torch.nn.BCELoss())
    disc_optim = attr.ib(None)
    gen_optim = attr.ib(None)
    device = attr.ib(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    epochs_trained = attr.ib(0, init=False)
    epoch_cb_history = attr.ib(attr.Factory(list), init=False)
    batch_cb_history = attr.ib(attr.Factory(list), init=False)

    default_optim_cls = torch.optim.Adam
    #default_optim_cls_kws =

    def __attrs_post_init__(self):
        self.model_map = {k: v.to(self.device) for k, v in self.model_map.items()}
        #self.opt_map = dict()
        for k, m in self.model_map.items():
            m.apply(self.weights_init)

            if k not in self.opt_map:
                if self.default_optim_cls == torch.optim.Adam:
                    self.opt_map[k] = self.default_optim_cls(m.parameters(),
                                                       lr=self.learning_rate,
                                                       betas=(self.beta1, 0.999))
                elif self.default_optim_cls == torch.optim.RMSprop:
                    self.opt_map[k] = self.default_optim_cls(m.parameters(),
                                                             lr=self.learning_rate)


    #            self.opt_map[k] = torch.optim.Adam(m.parameters(),
    #                                               lr=self.learning_rate,
    #                                               betas=(self.beta1, 0.999))



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
            #single_dims = [1] * (self.in_channel_dim - 1)
            # noise = torch.randn(batch_size, self.in_channel_size,
            #                    *single_dims,
            #                    device=self.device)
            size = [batch_size, self.in_channel_size] + list(self.in_trailing_dims)
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
        self.epoch_results = getattr(self, 'epoch_results', dict(epoch=list(), batch=list()))

        self.epoch_cb_history += [{k: cb(self, 0) for k, cb in epoch_callbacks.items()}]
        self.batch_cb_history += [{k: cb(self, 0) for k, cb in batch_callbacks.items()}]

        if self.n_samples is None:
            try:
                self.n_samples = len(self.data_gen)
            except:
                print("Unable to determine generator lengthe using __len__")

        with tqdm(total=n_epochs,
                  desc='Training epoch',
                  dynamic_ncols=True
                  #ncols='100%'
                  ) as epoch_pbar:
            for epoch in range(self.epochs_trained, self.epochs_trained + n_epochs):
                with tqdm(total=self.n_samples, desc='-loss-', dynamic_ncols=True) as batch_pbar:
                    for i, data in enumerate(self.data_gen):
                        update_d = self.train_inner_step(epoch, data)

                        self.epoch_results['epoch'].append(epoch)
                        self.epoch_results['batch'].append(i)

                        prog_msgs = list()
                        for k, v in update_d.items():
                            # TODO: What about spruious results? Maybe do list of dicts instead?
                            if k not in self.epoch_results:
                                self.epoch_results[k] = [v]
                            else:
                                self.epoch_results[k].append(v)


                            v_l = np.round(np.mean(self.epoch_results[k][-20:]), 4)
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
                            self.batch_cb_history.append({k: cb(self, epoch)
                                                     for k, cb in batch_callbacks.items()})

                #self.epoch_losses.append(dict(gen_losses=G_losses, disc_losses=D_losses))
                #self.epoch_losses.append(epoch_results)
                #self.train_batch_results.append(epoch_results)
                self.epochs_trained += 1
                self.epoch_cb_history.append({k: cb(self, epoch) for k, cb in epoch_callbacks.items()})
                epoch_pbar.update(1)
        #return self.train_batch_results
        return self.epoch_results

    @staticmethod
    def grid_display(data, figsize=(10, 10),
                     pad_value=0, title='',
                     xlabel='', ylabel='',
                     nrow=8, normalize=True, ax=None):
        from torchvision import utils as vutils
        from matplotlib import pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        ax.axis("off")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.imshow(np.transpose(vutils.make_grid(data,
                                                padding=2,
                                                nrow=nrow,
                                                normalize=normalize,
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
                      pad_value=0, ax=None):
        if batch_data is not None:
            fake_batch = batch_data
        else:
            # Generate fake image batch with G
            fake_batch = self.generate(noise=noise, batch_size=batch_size)

        return self.grid_display(fake_batch, figsize=figsize, title=title, pad_value=pad_value, ax=ax)


@attr.s
class EncGANTrainer(BaseTrainer):
    z_size = attr.ib(100)

    def train_inner_step(self, epoch_i, data_batch):
        real_label, fake_label = 1, 0
        real_labels, fake_labels = None, None

        disc_model = self.model_map['disc'].train()
        gen_model = self.model_map['gen'].train()
        enc_model = self.model_map['enc'].train()

        disc_optim = self.opt_map['disc']
        gen_optim = self.opt_map['gen']
        enc_optim = self.opt_map['enc']

        disc_model.zero_grad()

        # Take a real batch
        real_x = data_batch[0].to(self.device)
        batch_size = real_x.shape[0]
        label_shape = (batch_size,)

        ########
        # Discriminator updating
        ###
        # Labels for the real batch in disc
        if real_labels is None:
            real_labels = torch.full(label_shape, real_label,
                                     device=self.device,
                                     dtype=torch.float32).view(-1)

        ####
        # REAL Batch
        disc_real_output = disc_model(real_x).view(-1)
        d_real_err = self.criterion(disc_real_output, real_labels)
        real_z = enc_model(real_x)
        # real_z = torch.randn(batch_size, z_size, 1, 1, device=self.device)

        # (d_real_err + enc_loss).backward()

        ####
        # Fake Batch
        # Generator takes encodings as input
        # real_z = self.enc_model(disc_intermediate_output)

        # Generate images from encoded z
        gen_real_z_output = gen_model(real_z.view(batch_size,
                                                       self.z_size,
                                                       1, 1))
        # Disc binary classification of fake inputs
        if fake_labels is None:
            fake_labels = torch.full(label_shape,
                                     fake_label,
                                     device=self.device,
                                     dtype=torch.float32).view(-1)

        disc_fake_output = disc_model(gen_real_z_output).view(-1)
        d_fake_err = self.criterion(disc_fake_output, fake_labels)
        # d_fake_err.backward(retain_graph=True)
        d_err = d_real_err + d_fake_err  # + enc_loss
        d_err.backward(retain_graph=True)
        # d_real_err.backward()
        # d_fake_err.backward()

        disc_optim.step()

        #####
        # Generator updating
        ###
        # Give noise to G, then check that it is labeled fake
        gen_model.zero_grad()
        enc_model.zero_grad()
        real_z = enc_model(real_x)
        # real_z = torch.randn(batch_size, z_size, 1, 1, device=self.device)
        gen_real_z_output = gen_model(real_z.view(batch_size,
                                                       self.z_size,
                                                       1, 1))
        disc_fake_output = disc_model(gen_real_z_output).view(-1)

        # Calculate an error for the generator (e.g. did we trick discrim)
        g_d_err = self.criterion(disc_fake_output, real_labels)

        BCE = torch.nn.functional.binary_cross_entropy(gen_real_z_output,
                                                       real_x,
                                                       reduction='mean')
        # enc_loss = torch.norm(real_z) * 0.001
        enc_loss = BCE
        (g_d_err + BCE).backward(retain_graph=True)

        gen_optim.step()

        #self.gen_losses.append(g_d_err.item())
        #self.disc_losses.append(d_err.item())
        #self.enc_losses.append(enc_loss.item())
        return dict(g_d_err=g_d_err.items(), d_err=d_err.item(), enc_err=enc_loss.item())
