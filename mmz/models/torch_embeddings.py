import torch


def embed_modules_from_token_map(token_map, sizes=2):
    if isinstance(sizes, int):
        sizes = {k: sizes for k in token_map}
    elif isinstance(sizes, dict):
        pass

    embed_modules_d = {k: torch.nn.Embedding(len(d), sizes[k])
                       for k, d in token_map.to_dict().items()}
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