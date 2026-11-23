"""The cache key of the training samples must not change unless SIM_VERSION does: it names the stored labels and
the worker jobs. It hashes the assembly, the pipeline settings, the coupling settings and the stack file's text."""

import yaml

from geocouple.data.build import PipelineSpec
from geocouple.data.sampling import full_assemblies
from geocouple.data.simulate import SIM_VERSION, cache_key
from geocouple.graph.coupling import CouplingSpec
from geocouple.stack import load_stack

# data/train_v1 (sim_version p2.3), labels.json "cache_key"
KNOWN = {"bar_0003": "4c4829949c95d4cd"}


def test_cache_keys_of_the_training_set_are_stable():
    cfg = yaml.safe_load(open("configs/datasets/train.yaml"))
    stack = load_stack(cfg["stack"])
    spec, coupling = PipelineSpec.from_config(cfg["pipeline"]), CouplingSpec(**cfg["coupling"])
    asms = {a.name: a for a in full_assemblies(cfg, stack) if a.name in KNOWN}
    assert SIM_VERSION == "p2.3", "SIM_VERSION changed: update KNOWN from the new labels"
    for name, key in KNOWN.items():
        assert cache_key(asms[name], spec, coupling, stack) == key, name
