"""The scripts that reproduce figures quoted in the docs still run."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up while the class is built
    spec.loader.exec_module(module)
    return module


async def test_measure_agents_runs_on_a_small_corpus() -> None:
    # docs/agents.md section 7 comes from this script with --papers 200; four
    # papers run every strategy and both batch configurations in seconds.
    measure_agents = _load("measure_agents")
    results = await measure_agents.measure(4)
    table = measure_agents.markdown(results)
    for strategy in ("agentic", "fulltext", "mapreduce"):
        assert results["strategies"][strategy]["stages"]["read"]["papers"] == 4
        assert f"| {strategy} |" in table
    assert set(results["modes"]) == {"online", "batch, online settings", "batch, batch_settings"}
