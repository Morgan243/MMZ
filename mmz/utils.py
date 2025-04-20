from collections import namedtuple
import sys
import torch
from torch.autograd import Variable, Function
from typing import List, Callable, Dict, Optional, Tuple, Type, Any
from sklearn.metrics import classification_report
import subprocess
import json
import logging
import numpy as np
import torch.nn as nn
from graphviz import Digraph
import argparse


# https://stackoverflow.com/questions/42033142/is-there-an-easy-way-to-check-if-an-object-is-json-serializable-in-python
def is_jsonable(x: str) -> bool:
    """
    Check if an object is JSON serializable.

    Parameters
    ----------
    x : str
        The object to check for JSON serialization.

    Returns
    -------
    bool
        True if the object is JSON serializable, False otherwise.
    """
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def run_subprocess(command: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Run a subprocess and capture its output and error.

    Parameters
    ----------
    command : List[str]
        The command to run as a list of arguments.

    Returns
    -------
    Tuple[Optional[str], Optional[str]]
        A tuple containing the standard output and standard error.
    """
    try:
        result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = result.communicate()
        return output.decode('utf-8'), error.decode('utf-8') if error else None
    except FileNotFoundError:
        return None, "Executable not found."


def get_logger(logname: str = 'mmz', console_level: int = logging.DEBUG, file_level: int = logging.DEBUG,
               format_string: str = '%(asctime)s - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s',
               output_file: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with specified settings.

    Parameters
    ----------
    logname : str, optional
        The name of the logger.
    console_level : int, optional
        The logging level for the console handler.
    file_level : int, optional
        The logging level for the file handler.
    format_string : str, optional
        The format string for the log messages.
    output_file : Optional[str], optional
        The file path to write logs to.

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    logger = logging.getLogger(logname)
    if logger.hasHandlers():
        return logging.getLogger(logname)
    else:
        pass
        #print("MAKING NEW LOGGER: " + logname)
    #logger = < create_my_logger > if not logging.getLogger().hasHandlers() else logging.getLogger()
    # create logger with 'spam_application'
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(format_string)

    # create file handler which logs even debug messages
    if output_file is not None:
        fh = logging.FileHandler(output_file)
        fh.setLevel(file_level)
        logger.addHandler(fh)
        fh.setFormatter(formatter)

    # create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    # create formatter and add it to the handlers
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)
    return logger


def with_logger(cls: Optional[Type] = None, prefix_name: Optional[str] = None) -> Type:
    """
    Decorator to add a logger to a class.

    Parameters
    ----------
    cls : Optional[Type], optional
        The class to decorate.
    prefix_name : Optional[str], optional
        The prefix name for the logger.

    Returns
    -------
    Type
        The decorated class with a logger.
    """
    def _make_cls(cls):
        n = __name__ if prefix_name is None else prefix_name
        cls.logger = get_logger(n + '.' + cls.__name__)
        return cls

    cls = _make_cls if cls is None else _make_cls(cls)

    return cls


def iter_graph(root: Function, callback: Callable[[Function], None]) -> None:
    """
    Iterate over the computational graph starting from a root function.

    Parameters
    ----------
    root : Function
        The root function of the computational graph.
    callback : Callable[[Function], None]
        A callback function to apply to each node in the graph.
    """
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


def register_hooks(var: Variable) -> Callable[[], Digraph]:
    """
    Register hooks to capture gradients and build a computational graph.

    Parameters
    ----------
    var : Variable
        The variable to register hooks for.

    Returns
    -------
    Callable[[], Digraph]
        A function that returns the built computational graph.
    """
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
        from graphviz import Digraph
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

# Is there a built in way to do this on a class?
class SetParamsMixIn:
    """Adds a `set_params()` method to set attributes from a mapping, but only attributes that already exist.
    Otherwise, throws error"""
    def set_params(self, d: Optional[Dict] = None, **kws):
        if d is not None and isinstance(d, dict):
            p_d = d
        elif d is None:
            p_d = dict()
        else:
            raise ValueError()

        p_d.update(kws)

        for k, v in p_d.items():
            assert hasattr(self, k), f"{k} not an attribute of {self}"
            current_value = getattr(self, k)
            if issubclass(type(current_value), SetParamsMixIn) and isinstance(v, dict):
                current_value.set_params(v)
                setattr(self, k, current_value)
            else:
                setattr(self, k, v)

        return self


    @staticmethod
    def _set_recursive_dot_attribute(parent, dot_k: str, v: object, sep='.'):
        k_l = dot_k.split(sep)
        assert hasattr(parent, k_l[0])
        if len(k_l) == 1:
            setattr(parent, k_l[0], v)
        else:
            _next = getattr(parent, k_l[0])
            SetParamsMixIn._set_recursive_dot_attribute(_next, ".".join(k_l[1:]), v, sep=sep)
        return parent

    def set_recursive_dot_attribute(self, dot_k: str, v: object, sep='.', parent=None):
        self._set_recursive_dot_attribute(self if parent is None else parent,
                                         dot_k, v, sep)
        return self

    @staticmethod
    def _get_recursive_dot_attribute(parent, dot_k, sep='.'):
        k_l = dot_k.split(sep)
        #parent = self if parent is None else parent
        assert hasattr(parent, k_l[0])
        if len(k_l) == 1:
            v = getattr(parent, k_l[0])
        else:
            _next = getattr(parent, k_l[0])
            v = SetParamsMixIn._get_recursive_dot_attribute(_next, ".".join(k_l[1:]), sep=sep)
        return v

    def get_recursive_dot_attribute(self, dot_k, sep='.', parent=None):
        return self._get_recursive_dot_attribute(self if parent is None else parent,
                                                 dot_k, sep)


def example_grad_check_usage() -> None:
    """
    Example usage of gradient checking and graph visualization.
    """
    x = torch.randn(10, 10, requires_grad=True)
    y = torch.randn(10, 10, requires_grad=True)

    z = x / (y * 0)
    z = z.sum() * 2
    get_dot = register_hooks(z)
    z.backward()
    dot = get_dot()
    dot.save('tmp.dot')

def print_sequential_arch(m: nn.Sequential, t_x: torch.Tensor) -> None:
    """
    Print the architecture of a sequential model and intermediate predictions.

    Parameters
    ----------
    m : nn.Sequential
        The sequential model to inspect.
    t_x : torch.Tensor
        The input tensor to pass through the model.
    """
    for i in range(0, len(m)):
        print(m[i])
        l_preds = m[:i + 1](t_x)
        print(l_preds.shape)
        print("----")

def number_of_model_params(m: nn.Module, trainable_only: bool = True) -> int:
    """
    Calculate the total number of parameters in a model.

    Parameters
    ----------
    m : nn.Module
        The model to inspect.
    trainable_only : bool, optional
        Whether to count only trainable parameters.

    Returns
    -------
    int
        The total number of parameters.
    """
    p_cnt = sum(p.numel() for p in m.parameters()
                if (p.requires_grad and trainable_only) or not trainable_only)
    return p_cnt

def build_default_options(default_option_kwargs: List[Dict], **overrides) -> namedtuple:
    """
    Build default options from keyword arguments and overrides.

    Parameters
    ----------
    default_option_kwargs : List[Dict]
        A list of dictionaries containing default option configurations.
    **overrides : Any
        Keyword arguments to override the default options.

    Returns
    -------
    namedtuple
        A named tuple with the built options.
    """
    opt_keys = [d['dest'].replace('-', '_')[2:]
                for d in default_option_kwargs]
    Options = namedtuple('Options', opt_keys)
    return Options(*[d['default'] if o not in overrides else overrides[o]
                     for o, d in zip(opt_keys, default_option_kwargs)])

def performance(y: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """
    Calculate performance metrics for binary classification.

    Parameters
    ----------
    y : np.ndarray
        The true labels.
    preds : np.ndarray
        The predicted probabilities or class labels.

    Returns
    -------
    Dict[str, float]
        A dictionary containing F1 score, accuracy, precision, and recall.
    """
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    return dict(f1=f1_score(y, preds),
                accuracy=accuracy_score(y, preds),
                precision=precision_score(y, preds),
                recall=recall_score(y, preds),
                )

def make_classification_reports(output_map: Dict[str, Dict], pretty_print: bool = True, threshold: Optional[float] = 0.5) -> Dict[str, str]:
    """
    Generate classification reports for multiple datasets.

    Parameters
    ----------
    output_map : Dict[str, Dict]
        A dictionary mapping dataset names to their actuals and predictions.
    pretty_print : bool, optional
        Whether to print the reports in a readable format.
    threshold : Optional[float], optional
        The threshold for binary classification (None for multiclass).

    Returns
    -------
    Dict[str, str]
        A dictionary mapping dataset names to their classification reports.
    """
    out_d = dict()
    for dname, o_map in output_map.items():
        if threshold is None:
            report_str = classification_report(o_map['actuals'], o_map['preds'].argmax(1))
        else:
            report_str = classification_report(o_map['actuals'], (o_map['preds'] > 0.5))
        if pretty_print:
            print("-"*10 + str(dname) + "-"*10)
            print(report_str)
        out_d[dname] = report_str
    return out_d

def multiclass_performance(y: np.ndarray, preds: np.ndarray, average: str = 'weighted') -> Dict[str, float]:
    """
    Calculate performance metrics for multiclass classification.

    Parameters
    ----------
    y : np.ndarray
        The true labels.
    preds : np.ndarray
        The predicted class labels.
    average : str, optional
        The averaging strategy for multiclass metrics.

    Returns
    -------
    Dict[str, float]
        A dictionary containing F1 score, accuracy, precision, and recall.
    """
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    return dict(f1=f1_score(y, preds, average=average),
                accuracy=accuracy_score(y, preds),
                precision=precision_score(y, preds, average=average),
                recall=recall_score(y, preds, average=average),
                )

def build_argparse(default_option_kwargs: List[Dict], description: str = '') -> argparse.ArgumentParser:
    """
    Build an ArgumentParser with default options.

    Parameters
    ----------
    default_option_kwargs : List[Dict]
        A list of dictionaries containing default option configurations.
    description : str, optional
        The description for the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The configured ArgumentParser.
    """
    import argparse
    parser = argparse.ArgumentParser(description=description)
    for _kwargs in default_option_kwargs:
        first_arg = _kwargs.pop('dest')
        parser.add_argument(first_arg, **_kwargs)

    return parser

import sys, os
import torch.distributed as dist
## From tourch tutorial: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
def setup(rank: int, world_size: int, init_file: Optional[str] = None) -> None:
    """
    Set up the distributed environment.

    Parameters
    ----------
    rank : int
        The rank of the current process.
    world_size : int
        The total number of processes.
    init_file : Optional[str], optional
        The initialization file for Windows platform.
    """
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

def cleanup() -> None:
    """
    Clean up the distributed environment.
    """
    dist.destroy_process_group()

""" Dataset partitioning helper """
class Partition(object):
    """
    A partition of data based on indices.

    Parameters
    ----------
    data : Any
        The data to partition.
    index : List[int]
        The list of indices defining the partition.
    """

    def __init__(self, data: Any, index: List[int]):
        self.data = data
        self.index = index

    def __len__(self) -> int:
        """
        Get the length of the partition.

        Returns
        -------
        int
            The number of elements in the partition.
        """
        return len(self.index)

    def __getitem__(self, index: int) -> Any:
        """
        Get an element from the partition by index.

        Parameters
        ----------
        index : int
            The index of the element to retrieve.

        Returns
        -------
        Any
            The element at the specified index.
        """
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
