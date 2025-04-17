from guidance import gen
from dataclasses import dataclass, field
from typing import Optional
from simple_parsing import Serializable

import pandas as pd
from simple_parsing import ArgumentParser
from tqdm.auto import tqdm

from mmz.agents.cheap_research import CheapResearch
import mlflow


@dataclass
class WebResearchAgent(Serializable):
    query_preset = 'simple'

    query_preset_map: dict[str, list[str]] = field(default_factory=lambda: {
        'simple': [
            'what years was trump president?',
            'what years was the cold war?',
            'what years was bill gates president?',
            'who was the first president of the United States?',
            'when did the civil war start and end?',
            'who wrote "1984"?',
            'what is the capital of Peru?',
            'how many planets are in our solar system?',
            'who painted the Mona Lisa?'
        ]
    })
    cheap_research: CheapResearch = field(default_factory=CheapResearch)
    queries: Optional[list[str]] = None

    def __post_init__(self):
        if self.queries is None:
            self.queries = self.query_preset_map[self.query_preset]

    def run(self):
        # setup and mlflow experiment
        mlflow.set_experiment("Web Research Agent")

        with mlflow.start_run():
            mlflow.log_params(self.cheap_research.to_dict())
            mlflow.log_dict(self.to_dict(), artifact_file='./experiment_config')

            results_l = list()
            for ii, q in enumerate(self.queries):
                print(f"Starting Query: {q}")
                self.cheap_research.query = q
                result = self.cheap_research.run()
                res_d = dict(cheap_research_config=self.cheap_research.to_dict(),
                             response=result)
                base_artifact_path = f'search_experiments/query_{ii}/'
                mlflow.log_table(self.cheap_research.results_frame_.T,
                                 artifact_file=f'{base_artifact_path}/literature_table.json')
                mlflow.log_dict(res_d, artifact_file=f'{base_artifact_path}/config.json')
                mlflow.log_text(result, artifact_file=f'{base_artifact_path}/result.txt')
                res_d['query'] = q
                res_d['query_ix'] = ii
                results_l.append(res_d)

            all_outputs_df = pd.DataFrame(results_l)
            mlflow.log_table(all_outputs_df, artifact_file=f'search_experiment_outputs.json')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(WebResearchAgent, dest="options")
    args = parser.parse_args()
    exper_cli: WebResearchAgent = args.options
    exper_cli.run()

