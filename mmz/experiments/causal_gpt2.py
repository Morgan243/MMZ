import os
from dataclasses import dataclass, field
import torch
from simple_parsing import Serializable
from contextlib import nullcontext

# Impotr a bunch of common typing types
from typing import Any, Callable, Dict, List, Optional, Tuple

from datasets import load_dataset, DatasetDict
from transformers import GPT2Tokenizer

from pytorch_lightning import LightningModule, Trainer
import torch


class LitModel(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


@dataclass
class TextDataset(Serializable):
    train_text: Optional[str] = None
    val_text: Optional[str] = None

    encoded_path: str = "/home/morgan/Projects/llm_poc/datasets/roneneldan/TinyStories/tiny_stories.hf"
    cache_dir: str = "/home/botbag/cache"

    def __post_init__(self):
        pass

    def load_data(self) -> DatasetDict:
        return load_dataset("text",
                            streaming=True,
                            num_proc=8,
                            data_files={'train': self.train_text,
                                        'test': self.val_text})

    def initialize_tokenizer(self) -> GPT2Tokenizer:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer

    def encode_data(self, dataset: DatasetDict,
                    tokenizer: GPT2Tokenizer,
                    num_proc: int = 8) -> DatasetDict:
        def encode_function(examples):
            return tokenizer(examples['text'], return_tensors="np",
                             padding='max_length', truncation=True)

        return dataset.map(encode_function, batched=True, num_proc=num_proc)

    @property
    def encoded_dataset(self) -> DatasetDict:
        if os.path.exists(self.encoded_path):
            # Load the pre-encoded dataset
            encoded_dataset = load_dataset(self.encoded_path, streaming=True,
                                           cache_dir=self.cache_dir)
        else:
            assert False
            # Load the raw data and encode it
            dataset = self.load_data()
            tokenizer = self.initialize_tokenizer()
            encoded_dataset = self.encode_data(dataset, tokenizer)

            if self.encoded_dataset is not None:
                # Save the encoded dataset to disk for future use
                encoded_dataset.save_to_disk(self.encoded_path)

        return encoded_dataset

    @property
    def orig_encoded_dataset(self):
        dataset_dict = self.load_data()
        tokenizer = self.initialize_tokenizer()
        encoded_dataset = self.encode_data(dataset_dict, tokenizer)
        return encoded_dataset



def encode_tiny_stories():
    dataset = TextDataset(
        train_text="/home/morgan/Projects/llm_poc/datasets/roneneldan/TinyStories/TinyStories-train.txt",
        val_text="/home/morgan/Projects/llm_poc/datasets/roneneldan/TinyStories/TinyStories-valid.txt"
    )

    dataset_dict = dataset.load_data()
    tokenizer = dataset.initialize_tokenizer()
    encoded_dataset = dataset.encode_data(dataset_dict, tokenizer)
    encoded_dataset.save_to_disk("/home/morgan/Projects/llm_poc/datasets/roneneldan/TinyStories/tiny_stories.hf")


@dataclass
class CausalGPT2Pretraining(Serializable):
    """
    """
    out_dir: str = 'out'
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = 'scratch'  # 'scratch', 'resume', or 'gpt2*'

    #wandb_log: bool = False  # disabled by default
    #wandb_project: str = 'owt'
    #wandb_run_name: str = 'gpt2'  # 'run' + str(time.time())

    dataset: TextDataset = field(default_factory=TextDataset)
    gradient_accumulation_steps: int = 5 * 8  # used to simulate larger batch sizes
    batch_size: int = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size: int = 1024

    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False  # do we use bias inside LayerNorm and Linear layers?

    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 600000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    backend: str = 'nccl'  # 'nccl', 'gloo', etc.

    device: str = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
    # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = True  # use PyTorch 2.0 to compile the model to be faster

    @classmethod
    def get_model_kws(cls, config: 'CausalGPT2Pretraining') -> Dict[str, Any]:
        # Just remap these to access the attribute off this
        # methods config parameter
        model_args = dict(vocab_size=None,
                          n_layer=config.n_layer,
                          n_head=config.n_head,
                          n_embd=config.n_embd,
                          dropout=config.dropout,
                          bias=config.bias,
                          block_size=config.block_size)

        meta_vocab_size = None
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304

        return model_args

    @classmethod
    def run(cls, config: 'CausalGPT2Pretraining'):
        import torch
        print("running")
        seed_offset = 0
        torch.manual_seed(1337 + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        device_type = 'cuda' if 'cuda' in config.device else 'cpu'  # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        # poor man's data loader
        #data_dir = os.path.join('data', dataset)
        from mmz.models.torch_gpt2 import GPT, GPTConfig
        model_config = GPTConfig(**cls.get_model_kws(config))
        model = GPT(model_config)

        # crop down the model block size if desired, using model surgery
        if config.block_size < model.config.block_size:
            model.crop_block_size(config.block_size)
            model_config['block_size'] = config.block_size  # so that the checkpoint will have the right value
        model.to(config.device)

        scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))

        # optimizer
        optimizer = model.configure_optimizers(config.weight_decay,
                                               config.learning_rate,
                                               (config.beta1, config.beta2),
                                               device_type)

        # compile the model
        if config.compile:
            print("compiling the model... (takes a ~minute)")
            unoptimized_model = model
            model = torch.compile(model) # requires PyTorch 2.0

        # Assuming model and config are already defined
        lit_model = LitModel(model)
        trainer = Trainer(max_epochs=1)
        trainer.fit(lit_model, config.dataset.encoded_dataset)

    def __call__(self):
        self.__class__.run(self)


config = CausalGPT2Pretraining()
config()

