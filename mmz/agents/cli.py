from dataclasses import dataclass, field
from simple_parsing import Serializable
from simple_parsing import ArgumentParser
from mmz.agents.cheap_research import CheapResearch


@dataclass
class AgentsCLI(Serializable):
    cheap_research: CheapResearch = field(default_factory=CheapResearch)
    #query: str = "watergate tape transcripts"
    #guidance_guide: GuidanceGuide = field(default_factory=GuidanceGuide)

    def run(self):
        results = self.cheap_research.run()
        #print("Guidance Guide")
        #print(self.guidance_guide)
        #results = cheap_research(self.query, self.guidance_guide)
        #print("Cheap research complete!")

        return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(AgentsCLI, dest="options")
    args = parser.parse_args()
    agents_cli: AgentsCLI = args.options
    #print(agents_cli.cheap_research.gg)
    agents_cli.run()
    #del agents_cli.guidance_guide
    #del agents_cli

