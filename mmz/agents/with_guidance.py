
from dataclasses import dataclass, field
from guidance import models as gmodels
from mmz.agents import tools as mzt
from typing import ClassVar, Optional
import pydantic
from pydantic import create_model
from functools import cached_property
import json
import os

import guidance
from guidance import one_or_more, select, zero_or_more
from simple_parsing import Serializable
import numpy as np


@dataclass
class GuidanceLlamaCppConfig(Serializable):
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    n_gpu_layers: int = 0
    n_ctx: int = 1024

    loaded_model_: gmodels.Model = field(default=None, init=False)

    @classmethod
    def make_kws(cls, model_name: str, **overrides) -> dict[str, object]:
        from copy import deepcopy
        base_kws = deepcopy(cls.model_kws_map[model_name])
        base_kws.update(overrides)
        return base_kws

    @property
    def model(self) -> gmodels.Model:
        if self.loaded_model_ is None:
            self.loaded_model_ = gmodels.LlamaCpp(self.model_path, echo=False,
                                                 n_gpu_layers=self.n_gpu_layers,
                                                 n_ctx=self.n_ctx)
        return self.loaded_model_

    @classmethod
    def make_preset_map(cls):
        default_configs = {
            'small': GuidanceLlamaCppConfig(
                model_path='/home/botbag/external/hf/Qwen/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct-Q8_0.gguf',
                n_ctx=32000,
                n_gpu_layers=35,
            ),
            # This is very slow, maybe broken, on older GPUs (1070GTX)
            'med': GuidanceLlamaCppConfig(
                model_path='/home/botbag/external/hf/Qwen/Qwen2.5-3B-Instruct/Qwen2.5-3B-Instruct-Q8_0.gguf',
                n_ctx=32000,
                n_gpu_layers=30,
            ),
            'med_bf16': GuidanceLlamaCppConfig(
                # /home/botbag/external/hf/Qwen/Qwen2.5-3B-Instruct/Qwen2.5-3B-Instruct-BF16.gguf
                model_path='/home/botbag/external/hf/Qwen/Qwen2.5-3B-Instruct/Qwen2.5-3B-Instruct-BF16.gguf',
                n_ctx=32000,
                n_gpu_layers=18,
            ),
            'large': GuidanceLlamaCppConfig(
                # /home/botbag/external/hf/Qwen/Qwen2.5-3B-Instruct/Qwen2.5-3B-Instruct-BF16.gguf
                model_path='/home/botbag/external/hf/Qwen/Qwen2.5-7B-Instruct/Qwen2.5-7B-Instruct-BF16.gguf',
                n_ctx=32000,
                n_gpu_layers=10,
            ),

        }
        return default_configs

    @classmethod
    def get_preset(cls, name: str = 'small') -> 'GuidanceLlamaCppConfig':
        configs = cls.make_preset_map()
        return configs[name]


@dataclass
class GuidanceOpenAIOllamaConfig(Serializable):
    """
    Tried to do this to have guidance work with a local remote servers
    but this breaks...guidance must use soem openai specific api functionality here..
    """
    model_name: str = 'qwen2.5-coder:7b-instruct-q4_K_M'
    #base_url: str = 'http://mesh:8000/v1'
    base_url: str = 'http://mesh:11434/v1'
    #base_url: str = 'http://127.0.0.1:11434/'
    #local openai_compat_url = 'http://mesh:11434/v1/'


    loaded_model_: gmodels.Model = field(default=None, init=False)

    @property
    def model(self) -> gmodels.Model:
        if self.loaded_model_ is None:
            self.loaded_model_ = gmodels.OpenAI(model=self.model_name, api_key='ollama',
                                               base_url=self.base_url, echo=False)
        return self.loaded_model_

    def test(self):
        self = GuidanceOpenAIOllamaConfig()
        model = self.model

        from guidance.library._role import assistant
        with assistant():
            model + "hi"
            res = self.test()
            print(res)




