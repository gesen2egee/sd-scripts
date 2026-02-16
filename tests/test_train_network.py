import importlib.machinery
import sys
import types

try:
    import xformers.ops  # noqa: F401
except Exception:  # pragma: no cover - environment dependent optional deps
    xformers_stub = types.ModuleType("xformers")
    xformers_stub.__spec__ = importlib.machinery.ModuleSpec("xformers", loader=None)
    xformers_stub.__version__ = "0.0.0"
    xformers_ops_stub = types.ModuleType("xformers.ops")
    xformers_ops_stub.__spec__ = importlib.machinery.ModuleSpec("xformers.ops", loader=None)
    xformers_stub.ops = xformers_ops_stub
    sys.modules["xformers"] = xformers_stub
    sys.modules["xformers.ops"] = xformers_ops_stub

import train_network

def test_syntax():
    # Very simply testing that the train_network imports without syntax errors
    assert True
