import sys
from pathlib import Path


def _add_src_to_path() -> None:
    # Ensure 'src' package is importable as 'src.*' by adding project root
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_add_src_to_path()


