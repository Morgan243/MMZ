import os
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Any, Tuple
from simple_parsing import Serializable
from functools import cached_property
import subprocess
import json
from mmz.utils import run_subprocess


@dataclass
class ExploitDB(Serializable):
    root_directory: str
    search_sploit_path: str = field(default=None, init=None)

    def __post_init__(self):
        self.search_sploit_path = os.path.join(self.root_directory, 'searchsploit')

    def list_exploits(self) -> List[str]:
        """List all exploits in the ExploitDB directory."""
        import os
        return [f for f in os.listdir(self.root_directory) if os.path.isfile(os.path.join(self.root_directory, f))]

    @cached_property
    def help_string(self):
        """Get the help string from searchsploit."""
        command = [self.search_sploit_path, '-h']
        output, error = run_subprocess(command)
        if error:
            return f"Error: {error}"
        else:
            return output

    def print_help(self) -> str:
        """Invoke searchsploit with --help argument using popen."""
        print(self.help_string)
        return self.help_string

    def searchsploit(self, *args: str) -> str:
        command = [self.search_sploit_path] + list(args)
        output, _ = run_subprocess(command)
        return output

    def searchsploit_as_json(self, *args: str,
                             deserialize_results: bool = True) -> str | Any:
        if not any(_j in args for _j in ['-j', '--json']):
            args = list(['-j'] + list(args))
        o = self.searchsploit(*args)
        o = json.loads(o) if deserialize_results else o
        return o

    @staticmethod
    def flatten_cve_results(results) -> list[dict]:
        """
            Flattens the given CVE results to a simple list of dictionaries.
            
            Args:
                results (dict): A dictionary containing CVE search results.
                
            Returns:
                list: A flattened list of dictionaries with high-level summaries of each CVE.
        """
        flattened_results = []

        # Flatten RESULTS_EXPLOIT
        for item in results['RESULTS_EXPLOIT']:
            flattened_item = {
                'Title': item['Title'],
                'EDB-ID': item['EDB-ID'],
                'Date_Published': item['Date_Published'],
                'Author': item['Author'],
                'Type': item['Type'],
                'Platform': item['Platform'],
                'Verified': item['Verified'],
                'Source': 'exploit-db:exploit'
            }
            flattened_results.append(flattened_item)

        # Flatten RESULTS_SHELLCODE
        for item in results['RESULTS_SHELLCODE']:
            flattened_item = {
                'Title': item['Title'],
                'EDB-ID': item['EDB-ID'],
                'Date_Published': item['Date_Published'],
                'Author': item['Author'],
                'Type': item['Type'],
                'Platform': item['Platform'],
                'Verified': item['Verified'],
                'Source': 'exploit-db:shellcode'
            }
            flattened_results.append(flattened_item)

        return flattened_results

    def searchsploit_as_summary(self, *args: str,
                                fields: list[str] = None,
                                n: int = None) -> str:
        fields = ['index', 'Title'] if fields is None else fields
        json_dat = self.searchsploit_as_json(*args)
        df = pd.DataFrame(self.flatten_cve_results(json_dat))
        _df = df.reset_index()[fields]
        _df = _df if n is None else _df.head(n)
        return _df.to_json()


import guidance
from guidance import capture, Tool
from pydantic import create_model
from guidance import regex

@guidance(stateless=True)
def searchsploit_call(lm):
    #schema = create_model(f"{name}", **{name: str, 'confidence': int})
    #json_qa = guidance.json(name=name, schema=schema)
    words_rx = regex(r'\w+')
    words_rx.match('foo bar')

    # capture just 'names' the expression, to be saved in the LM state
    return lm + 'searchsploit(' + capture(words_rx, 'tool_args') + ')'


@guidance
def searchsploit(lm):
    search_terms = lm['tool_args']
    # You typically don't want to run eval directly for security reasons
    # Here we are guaranteed to only have mathematical expressions
    #lm += f' = {eval(search_terms)}'
    #db = ExploitDB()
    db = ExploitDB(root_directory="/home/morgan/Projects/EXTERNAL/exploitdb/")
    lm += f' = {db.searchsploit_as_summary(*search_terms, n=10)}'
    return lm


def test_guide():
    #from mmz.agents.with_guidance import GuidanceLlamaCppConfig
    from mmz.agents.tools.with_guidance import GuidanceGuide
    from guidance import gen
    gg = GuidanceGuide()
    searchsploit_tool = Tool(searchsploit_call(), searchsploit)
    few_shot = 'You are on a linux 6.0 system, write a brief report about the vulnerabilities from CVEs'
    lm = gg.model + few_shot + gen(max_tokens=1000,
                                   tools=[searchsploit_tool],
                                   stop=')')
    print(lm)

test_guide()

#exploit_db = ExploitDB(root_directory="/home/morgan/Projects/EXTERNAL/exploitdb/")
#t = exploit_db.searchsploit_as_summary('linux', 'password')
#len(t)
#
#
##json_dat = exploit_db.searchsploit_as_json('dell')
#json_dat = exploit_db.searchsploit_as_json('linux', 'password')
#df = pd.DataFrame(exploit_db.flatten_cve_results(json_dat))
#df.reset_index()[['index', 'Title']].to_json()
#df.info()
#json_dat.keys()
#{k: type(v) for k, v in json_dat.items()}
#{k: v if isinstance(v, str) else v[:5] for k, v in json_dat.items()}

#type(raw_json['SEARCH'])
#exploit_db.print_help()

#if __name__ == "__main__":
    # Assuming root_directory is set to the correct path
#    exploit_db = ExploitDB(root_directory="/home/morgan/Projects/EXTERNAL/exploitdb/")
#    exploit_db.print_help()
