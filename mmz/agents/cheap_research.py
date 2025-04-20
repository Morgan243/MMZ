from guidance import gen
from dataclasses import dataclass, field
from typing import Optional
from simple_parsing import Serializable
import json

from mmz import agents as az
from mmz.agents import tools as mzt
from mmz.agents.tools import GuidanceGuide
from mmz.agents import external_sources as mxs
import pandas as pd
from tqdm.auto import tqdm


@dataclass
class CheapResearch(Serializable):
    """
    TODO:
    This agent aims to provide an answer in the fastest and cheapest way possible.
    - make more simple read only system access tools, like list process, list files
    """
    query: str = None
    gg: Optional[GuidanceGuide] = None
    gg_relevance: Optional[GuidanceGuide] = None
    relevance_strategy: str = 'score'
    auto: bool = True

    expansion_limit: int = 3
    results_frame_: pd.DataFrame = field(default=None, init=None)

    @staticmethod
    def relevance_by_independent_scoring(gg: GuidanceGuide,
                                         query: str,
                                         summaries: list) -> pd.DataFrame:
        scores = [dict(relevance_score=gg.get_relevance_score(user_q=query, summary=s),
                       summary=s, title=s['title'])
                  for ii, s in tqdm(enumerate(summaries),
                                    desc="Reviewing summaries",
                                    total=len(summaries))]
        scores_df = (pd.DataFrame(scores)
                     .sort_values('relevance_score', ascending=False))
        scores_df['relevance_score'] = scores_df['relevance_score'].astype(int)
        return scores_df

    def run(self):
        gg = self.gg = GuidanceGuide() if self.gg is None else self.gg
        query = self.query

        print("Expanding topics...")
        ex_topics = gg.expand_topics(query)
        print("Expanded topics: ")
        print(ex_topics)

        all_queries = [query]
        if self.expansion_limit > 0:
            all_queries += ex_topics[:self.expansion_limit]

        print("All queries: ")
        print(all_queries)

        s = mxs.WikipediaTwoPhaseSearch()
        titles = s.query_for(all_queries)
        summaries = s.get_summaries(titles)

        if self.relevance_strategy == 'cites':
            relevance_ixes = gg.get_relevant_ixes_from_summary(user_q=query,
                                                               summaries=summaries,
                                                               relevance_grammar=az.relevance_by_json_int_list)
            print(f"Num summaries = {len(summaries)}")
            input(f"Relevances = {relevance_ixes}, continue? :")
            ordered_content = [s.get_content(summaries[ii]['title']) for ii in relevance_ixes]
            filtered_content = ordered_content[:5]
        elif self.relevance_strategy == 'score':
            scores_df = self.relevance_by_independent_scoring(
                gg=self.gg, query=query, summaries=summaries)
            scores_df['is_relevant'] = scores_df.relevance_score.pipe(
                lambda s: s.gt(s.median()) | s.eq(s.max()))
            # display the full dataframwe, all columns and rows no matter the size
            pd.set_option('display.max_rows', None,
                          'display.max_columns', None)
            print(scores_df)
            #input(f"Scores {scores_df.relevance_score.values.tolist()}, continue?: ")
            ordered_content = scores_df.query("is_relevant").summary.tolist() #[summaries[ii] for ii in list(sorted(scores))]
            filtered_content = ordered_content
            self.results_frame_ = scores_df

        txt_res = json.dumps(filtered_content, indent=2)

        prompt = f"""Given this background content\n--------------{txt_res}------------\nAnswer this query concisely: {query}\n"""
        print(prompt)
        out = gg.model + prompt + mzt.get_q_and_a_grammar(name='answer') #gen("answer", max_tokens=1024)
        synth_text = out['answer']
        print("-----")
        print(synth_text)

        return synth_text


#gg = GuidanceGuide()
#print(gg.model_path)
#out = gg.expand_topics("donal trump")
#print(out)
#type(out)
#import json
#res = json.loads(out)
#res


