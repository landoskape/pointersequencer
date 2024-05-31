from argparse import ArgumentParser
from .encoding_representation import EncodingRepresentationAnalysis

ANALYSIS_REGISTRY = {
    "encoding_analysis": EncodingRepresentationAnalysis,
}


def _check_analysis(analysis_name):
    """check if analysis is in the registry"""
    if analysis_name not in ANALYSIS_REGISTRY:
        valid_analyses = list(ANALYSIS_REGISTRY.keys())
        raise ValueError(f"Analysis ({analysis_name}) is not in ANALYSIS_REGISTRY, valid analyses are: {valid_analyses}")


def get_analysis(analysis_name, build=False, **kwargs):
    """
    lookup analysis constructor from analysis registry by name

    if build=True, builds analysis and returns an analysis object using any kwargs
    otherwise just returns the class constructor
    """
    _check_analysis(analysis_name)
    analysis = ANALYSIS_REGISTRY[analysis_name]
    if build:
        return analysis(**kwargs)
    return analysis


def create_analysis():
    """
    method to create analysis using initial argument parser

    the argument parser looks for the first positional argument (called "analysis"), and the resulting
    string is used to retrieve an analysis constructor from the ANALYSIS_REGISTRY

    any remaining arguments (args) are passed to the analysis constructor which has it's
    own argument parser in the class definition

    note:
    add_help=False so adding a --help argument will show a help message for the specific analysis
    that is requested rather than showing a help message for this little parser then blocking the
    rest of the execution. It means using --help requires a valid 'analysis' positional argument.
    """
    parser = ArgumentParser(description=f"ArgumentParser for loading analysis constructor", add_help=False)
    parser.add_argument("analysis", type=str, help="a string that defines which experiment to run")
    analysis_args, args = parser.parse_known_args()
    return get_analysis(analysis_args.analysis, build=True, args=args)
