import numpy as np
import torch
from torch import nn
import attr

import attr
from tqdm.auto import tqdm


class VEEGAN(nn.Module):
    def __init__(self, z_size, input_dim):
        super(VEEGAN, self).__init__()

        self.gen_m = nn.Sequential(
            #nn.
        )


@attr.attrs
class VEEGAN_Trainer:
    gen_m = attr.ib()
    disc_m = attr.ib()
    enc_m = attr.ib()

    data_gen = attr.ib()

    #in_channel_size = attr.ib(100)
    #in_channel_dim = attr.ib(3)
    z_size = attr.ib(5)
    z_mu = attr.ib(0)
    z_var = attr.ib(0.5)
    input_reshape = attr.ib(None)
    n_samples = attr.ib(None)
    learning_rate = attr.ib(0.0003)
    beta1 = attr.ib(0.5)
    criterion = attr.ib(torch.nn.BCELoss())
    disc_optim = attr.ib(None)
    gen_optim = attr.ib(None)
    enc_optim = attr.ib(None)
    device = attr.ib(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    epochs_trained = attr.ib(0, init=False)


    def train(self, n_epochs=1):
        total_batches = len(self.data_gen) if self.n_samples is None else self.n_samples
        def maybe_reshape(_arr, s):
            if s is None:
                return _arr
            assert isinstance(s, tuple)
            return _arr.reshape(*s)

        for n in tqdm(range(n_epochs)):
            with tqdm(total=total_batches, desc='epoch ' + str(n)) as batch_bar:
                for batch_i, (real_x, real_y) in enumerate(self.data_gen):
                    self.gen_optim.zero_grad()
                    self.disc_optim.zero_grad()
                    self.enc_optim.zero_grad()

                    ## Real Sample
                    fl_real_x = maybe_reshape(real_x, self.input_reshape)
                    ## Gaussian Z
                    z_i = torch.empty(fl_real_x.shape[0], self.z_size).normal_(mean=self.z_mu, std=self.z_var)

                    ## G Sample from Gaussian Z
                    g_from_z = self.gen_m(z_i)
                    ## Enc Z from G Sample
                    z_i_g = self.enc_m(g_from_z)

                    ###
                    ## formula 7
                    # disc_g_out = disc_m(torch.cat([z_i, g_from_z], dim=1))
                    disc_g_out = -torch.log(self.disc_m(torch.cat([z_i, g_from_z], dim=1).clamp(1e-8)))
                    #disc_g_out = torch.sigmoid(self.disc_m(torch.cat([z_i, g_from_z], dim=1)))
                    d_z = torch.sqrt((z_i - z_i_g) ** 2).mean()

                    o_hat = (disc_g_out.mean() + d_z)
                    o_hat.backward()
                    self.enc_optim.step()
                    self.gen_optim.step()

                    ###########
                    ## Enc of real
                    ## Real Sample
                    fl_real_x = maybe_reshape(real_x, self.input_reshape)
                    ## Gaussian Z
                    z_i = torch.empty(fl_real_x.shape[0], self.z_size).normal_(mean=self.z_mu, std=self.z_var)

                    ## G Sample from Gaussian Z
                    g_from_z = self.gen_m(z_i)

                    ## Enc Z from G Sample
                    z_i_g = self.enc_m(g_from_z)

                    disc_fake_out = self.disc_m(torch.cat([z_i, g_from_z], dim=1))
                    disc_real_out = self.disc_m(torch.cat([z_i_g, fl_real_x], dim=1))

                    ## Formula 8
                    #o_lr = -(torch.log(disc_fake_out) + torch.log(1. - disc_real_out)).mean()
                    o_lr = -(torch.log(disc_real_out.clamp(1e-8)) + torch.log(1. - disc_fake_out.clamp(0., 0.999))).mean()

                    o_lr.backward()

                    #optim_omega.step()
                    self.disc_optim.step()

                    # desc = "%.3f %.3f %.3f" % (g_w, g_theta, g_gamma)
                    #desc = "%.3f %.3f %.3f" % (o_hat, o_lr, d_z)
                    desc = "g=%.3f;z=%.3f;d=%.3f" % (o_hat - d_z, d_z, o_lr)
                    if any([torch.isnan(o_hat), torch.isnan(d_z), torch.isnan(o_lr)]):
                        print("NULL")
                        print(desc)
                        return
                    batch_bar.set_description(desc)
                    batch_bar.update(1)

        base_plt_i = 5
        n_to_plt = 5
        import matplotlib
        import seaborn as sns
        real_encs = self.enc_m(fl_real_x)
        g_from_real_z = self.gen_m(real_encs)
        fig, axs = matplotlib.pyplot.subplots(figsize=(13, 6), nrows=2, ncols=n_to_plt)
        for i in range(n_to_plt):
            _i = base_plt_i + i
            sns.heatmap((fl_real_x[_i].reshape(28, 28)).float().detach().numpy(),
                        ax=axs[0][i], cbar=False)

            sns.heatmap((g_from_real_z[_i].reshape(28, 28)).float().detach().numpy(),
                        ax=axs[1][i], cbar=False)
        fig.tight_layout()