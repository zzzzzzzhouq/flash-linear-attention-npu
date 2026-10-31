"""Load the runtime GDN timer layout without relying on a fixed repo depth."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_runtime_layout():
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "torch_custom/fla_npu/fla_npu/ops/ascendc/gdn_timer_layout.py"
        if candidate.is_file():
            spec = spec_from_file_location("_gdn_runtime_timer_layout", candidate)
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    from fla_npu.ops.ascendc import gdn_timer_layout
    return gdn_timer_layout


_LAYOUT = _load_runtime_layout()
globals().update({name: value for name, value in vars(_LAYOUT).items() if not name.startswith("_")})
