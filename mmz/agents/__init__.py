from dataclasses import dataclass, field
from guidance import models, gen, block
from mmz.agents import tools as mzt
from typing import ClassVar, Optional
import pydantic
from functools import cached_property
import json
import os

import guidance
from guidance import one_or_more, select, zero_or_more
from simple_parsing import Serializable
import numpy as np

#@dataclass
#class GuidanceLlamaCppConfig(Serializable):
#    model_name: Optional[str] = None
#    model_path: Optional[str] = None
#    n_gpu_layers: int = 0
#    n_ctx: int = 1024
#
#    loaded_model_: models.Model = field(default=None, init=False)
#
#    @classmethod
#    def make_kws(cls, model_name: str, **overrides) -> dict[str, object]:
#        from copy import deepcopy
#        base_kws = deepcopy(cls.model_kws_map[model_name])
#        base_kws.update(overrides)
#        return base_kws
#
#    @property
#    def model(self) -> models.Model:
#        if self.loaded_model_ is None:
#            self.loaded_model_ = models.LlamaCpp(self.model_path, echo=False,
#                                                 n_gpu_layers=self.n_gpu_layers,
#                                                 n_ctx=self.n_ctx)
#        return self.loaded_model_
#
#    @classmethod
#    def make_preset_map(cls):
#        default_configs = {
#            'small': GuidanceLlamaCppConfig(
#                model_path='/home/botbag/external/hf/Qwen/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct-Q8_0.gguf',
#                n_ctx=32000,
#                n_gpu_layers=15,
#            ),
#            'med': GuidanceLlamaCppConfig(
#                model_path='/home/botbag/external/hf/Qwen/Qwen2.5-3B-Instruct/Qwen2.5-3B-Instruct-Q8_0.gguf',
#                n_ctx=32000,
#                n_gpu_layers=30,
#            ),
#            'med_bf16': GuidanceLlamaCppConfig(
#                # /home/botbag/external/hf/Qwen/Qwen2.5-3B-Instruct/Qwen2.5-3B-Instruct-BF16.gguf
#                model_path='/home/botbag/external/hf/Qwen/Qwen2.5-3B-Instruct/Qwen2.5-3B-Instruct-BF16.gguf',
#                n_ctx=32000,
#                n_gpu_layers=18,
#            ),
#        }
#        return default_configs
#
#    @classmethod
#    def get_preset(cls, name:str = 'small') -> 'GuidanceLlamaCppConfig':
#        configs = cls.make_preset_map()
#        return configs[name]
#
#
## stateless=True indicates this function does not depend on LLM generations
#@guidance(stateless=True)
#def reference_selection(lm, n_total_refs, selection_name='selected_ixes'):
#    nums = [str(ii) for ii in range(n_total_refs)]
#    return lm + one_or_more(select(nums, name=selection_name))
#
#
##@guidance(stateless=True)
##def operator(lm):
##    return lm + select(['+' , '*', '**', '/', '-'])
#
#
#def get_summary_relevance_prompt(query: str, summaries: list[dict]) -> str:
#    import json
#    from datetime import datetime
#    current_time = datetime.now().strftime("%Y-%m-%d")
#    shuffled_ixes = list(range(len(summaries)))
#    np.random.shuffle(shuffled_ixes)
#    print("shuffled_ixes:", str(shuffled_ixes))
#    prompt = f"""Analyze these search results and provide a ranked list of the most relevant ones.
#
#IMPORTANT: Evaluate and rank based on these criteria (in order of importance):
#1. Timeliness - current/recent information as of {current_time}
#2. Direct relevance to query: "{query}"
#3. Source reliability (prefer official sources, established websites)
#4. Factual accuracy (cross-reference major claims)
#
#Search results to evaluate:
#  {json.dumps([{'reference index': ii,
#                'title': s['title'],
#                'summary': s['summary']}
#                for ii, s in enumerate(summaries)]
#              , indent=2)}
#
#Return ONLY a JSON array of the 0-based reference index, ranked from most to least relevant.
#Include ONLY indices that meet ALL criteria, with the most relevant first.
#You should list all {len(summaries)} indices in your response.
#You should not output any number larger than {len(summaries) - 1}
#Respond with ONLY the JSON array, no other text."""
##Example response (yours should be different!): {shuffled_ixes}
#    return prompt
#
#def get_summary_relevance_scalar_prompt(query: str, summary: dict):
#    import json
#    from datetime import datetime
#    current_time = datetime.now().strftime("%Y-%m-%d")
#    prompt = f"""Analyze these search results and provide a number
#between 0 and 100 according to its relevance to the users query,
#100 being the most relevant and likley answers the query
#0 being the least relevant and does not answer the query
#
#IMPORTANT: Evaluate and estimate relevance based on these criteria (in order of importance):
#1. Timeliness - current/recent information as of {current_time}
#2. Direct relevance to query: "{query}"
#
#Search results to evaluate:
#  {json.dumps({'title': summary['title'],
#               'summary': summary['summary']}
#              , indent=2)}
#
#Respond only with a number with in 0 and 100 and nothing else: """
#    return prompt

#@guidance
#def relevance_by_regex(llm, query, summaries):
#    relevance_prompt = get_summary_relevance_prompt(query, summaries=summaries)
#    out = llm + relevance_prompt + '[ ' + gen(regex=r'\d+') + ']'
#    return out
#
#
#@guidance
#def relevance_by_selection(llm, query, summaries, selection_name='selected_ixes'):
#    #print(f"Got summaries in relevance selection:\n{summaries}")
#    relevance_prompt = get_summary_relevance_prompt(query, summaries=summaries)
#    out = (llm + relevance_prompt
#        + '[ ' + reference_selection(n_total_refs=len(summaries),
#                                     selection_name=selection_name) + ']')
#    #print("Output produced")
#    return out
#
#
#def get_list_of_int_grammar(name="integers"):
#    from pydantic import create_model
#    schema = create_model(f"list_of_{name}", **{name: list[int]})
#    #class ListOfString(pydantic.BaseModel):
#    #    indices: list[str]
#    json_list = guidance.json(name=name, schema=schema)
#    return json_list
#
#
#def get_list_additional_topics_prompt(query: str) -> str:
#    prompt = """Given the users query, produce a JSON list of other topics related to their query.\n"""
#    prompt += f"""Here is their query: {query}\n"""
#    prompt += """Provide a list of JSON strings of related topics: """
#    return prompt
#
#
##@guidance
#def get_list_of_str_grammar(name="strings"):
#    from pydantic import create_model
#    schema = create_model(f"list_of_{name}", **{name: list[str]})
#    #class ListOfString(pydantic.BaseModel):
#    #    indices: list[str]
#    json_list = guidance.json(name=name, schema=schema)
#    return json_list


#@guidance
#def select_next(choices):
#    pass
#
#
#@guidance
#def relevance_by_json_int_list(llm, query, summaries, name='selected_ixes'):
#    relevance_prompt = get_summary_relevance_prompt(query, summaries=summaries)
#    #class ListOfIntegers(pydantic.BaseModel):
#    #    indices: list[int]
#    #json_list = guidance.json("selected_ixes", schema=ListOfIntegers)
#    #return llm + relevance_prompt + json_list
#    return llm + relevance_prompt + get_list_of_int_grammar(name=name)
#
#
#@guidance
#def relevance_scalar(llm, query, summary, name='relevance_magnitude'):
#    from pydantic import create_model
#    schema = create_model(f"scalar_{name}", **{name: int})
#    relevance_prompt = get_summary_relevance_scalar_prompt(query, summary=summary)
#    return llm  + relevance_prompt + guidance.json(name=name, schema=schema)
#
#
#@dataclass
#class GuidanceGuide(Serializable):
#    model_preset: Optional[str] = 'med'
#
#    model_config: Optional[GuidanceLlamaCppConfig] = None
#
#    def __post_init__(self):
#        if self.model_config is None:
#            self.model_config = GuidanceLlamaCppConfig.get_preset(self.model_preset)
#
#    @property
#    def model(self) -> models.Model:
#        return self.model_config.model
#
#    def get_relevant_ixes_from_summary(self, user_q: str,
#                                       summaries: list[dict],
#                                       relevance_grammar: callable = relevance_by_selection,
#                                       as_list: bool = True):
#        res = self.model + relevance_grammar(user_q, summaries=summaries)
#        res = res['selected_ixes']
#        if as_list:
#            if relevance_grammar == relevance_by_selection:
#                res = json.loads(f"[{res}]")
#            else:
#                res = json.loads(res['selected_ixes'])['selected_ixes']
#        return res
#
#    def filter_to_relevant_summeries(self, user_q:str, 
#                                     summaries: list[dict],
#                                     relevance_grammar: callable = relevance_by_selection) -> list[dict]:
#        ixes_to_keep = self.get_relevant_ixes_from_summary(
#            user_q=user_q,
#            summaries=summaries,
#            relevance_grammar=relevance_grammar
#        )
#        return [summaries[i] for i in ixes_to_keep]
#
#    def get_relevance_score(self, user_q:str, summary: dict) -> int:
#        res = self.model + relevance_scalar(query=user_q,
#                                            summary=summary)
#        res = json.loads(res['relevance_magnitude'])['relevance_magnitude']
#        return int(res)
#    
#
#    def expand_topics_g(self, user_q: str):
#        return (self.model
#            + get_list_additional_topics_prompt(query=user_q)
#            + get_list_of_str_grammar(name='topics'))
#
#    def expand_topics(self, user_q: str, as_list: bool=True) -> str |  list[str]:
#        # First ['topics'] access is to guidance to get that prompts raw results
#        res = self.expand_topics_g(user_q=user_q)['topics']
#        # Next access ['topics'] is to access the value at the 'topics' key to
#        # get the list of topics from the deserialized json
#        return json.loads(res)['topics'] if as_list else res
#
#    def answer_query(self, user_q: str, content):
#        return
#
#
