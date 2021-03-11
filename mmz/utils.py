from collections import namedtuple

from graphviz import Digraph
import torch
from torch.autograd import Variable, Function
import torch
from torch.autograd import Variable, Function

def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        if grad_output is None:
            return False
        return grad_output.isnan().any() or (grad_output.abs() >= 1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot

def example_grad_check_usage():
    x = torch.randn(10, 10, requires_grad=True)
    y = torch.randn(10, 10, requires_grad=True)

    z = x / (y * 0)
    z = z.sum() * 2
    get_dot = register_hooks(z)
    z.backward()
    dot = get_dot()
    dot.save('tmp.dot')

def print_sequential_arch(m, t_x):
    for i in range(0, len(m)):
        print(m[i])
        l_preds = m[:i + 1](t_x)
        print(l_preds.shape)
        print("----")

def number_of_model_params(m, trainable_only=True):
    p_cnt = sum(p.numel() for p in m.parameters()
                if (p.requires_grad and trainable_only) or not trainable_only)
    return p_cnt

def build_default_options(default_option_kwargs, **overrides):
    opt_keys = [d['dest'].replace('-', '_')[2:]
                for d in default_option_kwargs]
    Options = namedtuple('Options', opt_keys)
    return Options(*[d['default'] if o not in overrides else overrides[o]
                     for o, d in zip(opt_keys, default_option_kwargs)])

def performance(y, preds):
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    return dict(f1=f1_score(y, preds),
                accuracy=accuracy_score(y, preds),
                precision=precision_score(y, preds),
                recall=recall_score(y, preds),
                )

def build_argparse(default_option_kwargs, description=''):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    for _kwargs in default_option_kwargs:
        first_arg = _kwargs.pop('dest')
        parser.add_argument(first_arg, **_kwargs)

    return parser

import sys, os
import torch.distributed as dist
## From tourch tutorial: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
def setup(rank, world_size, init_file=None):
    if sys.platform == 'win32':
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        # Example init_method="file:///f:/libtmp/some_file"
        if init_file is None:
            raise ValueError("Init file path needed for windoz platform")
        init_method = "file:///" + str(init_file)

        # initialize the process group
        dist.init_process_group(
            "gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

""" Dataset partitioning helper """
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

# From https://pytorch.org/tutorials/intermediate/dist_tuto.html
from random import Random
class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])