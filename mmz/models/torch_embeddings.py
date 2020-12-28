import torch


def embed_modules_from_token_map(token_map, sizes=2):
    if isinstance(sizes, int):
        sizes = {k: sizes for k in token_map}
    elif isinstance(sizes, dict):
        pass

    embed_modules_d = {k: torch.nn.Embedding(len(d), sizes[k])
                       for k, d in token_map.items()}
    return embed_modules_d


def embed_modules_from_df(embed_df, sizes=3):
    em_map_s = embed_df.apply(lambda s: {v: i for i, v in enumerate(s.unique())},
                              result_type='reduce')
    return (embed_df.apply(lambda s: s.map(em_map_s[s.name])),
            em_map_s,
            embed_modules_from_token_map(em_map_s.to_dict(), sizes))


class MultiEmbed(torch.nn.Module):
    def __init__(self, embedding_layers, cat_dim=-1):
        super(MultiEmbed, self).__init__()
        self.embedding_layers = torch.nn.ModuleDict(embedding_layers)
        self.cat_dim = cat_dim

    def forward(self, x):
        return torch.cat([l(x.select(-1, i)).unsqueeze(self.cat_dim)
                          for i, l in enumerate(self.embedding_layers.values())],
                         dim=self.cat_dim)


class GloveBase(torch.nn.Module):
    x_max = 1
    alpha = 0.75

    def __init__(self, embedding_layers, bias_layers):
        super(GloveBase, self).__init__()
        self.embedding_layers = torch.nn.ModuleDict(embedding_layers)
        self.embedding_bias_layers = torch.nn.ModuleDict(bias_layers)

        assert len(self.embedding_layers) == len(self.embedding_bias_layers)

    @classmethod
    def weight_func(cls, x, x_max=None, alpha=None):
        x_max = cls.x_max if x_max is None else x_max
        alpha = cls.alpha if alpha is None else alpha

        weights = (x / x_max) ** alpha
        weights = torch.min(wx, torch.ones_like(weights, device=x.device))
        return weights

    @staticmethod
    def weighted_mse_loss(weights, inputs, targets):
        loss = weights * F.mse_loss(inputs, targets, reduction='none')
        return loss.mean()

    def forward(self, x):
        e_prod, b_sum = None, None
        for i, (emb, emb_bias) in enumerate(zip(self.embedding_layers.values(),
                                                self.embedding_bias_layers.values())):
            emb_codes = x.select(-1, i)
            e = emb(emb_codes)
            b = emb_bias(emb_codes).squeeze()

            e_prod = e if e_prod is None else e*e_prod
            b_sum = b if b_sum is None else b+b_sum

        interaction = e_prod.sum(-1) + b_sum
        return interaction

        #em_vec_arr = me_model(em_X_arr)
        #bias_em_vec_arr = bias_me_model(em_X_arr)

        #X_a, X_b = em_vec_arr.select(2, 0), em_vec_arr.select(2, 1)
        #bias_X_a, bias_X_b = bias_em_vec_arr.select(2, 0), bias_em_vec_arr.select(2, 1)
