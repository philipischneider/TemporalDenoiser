"""Detect and sort EXR frame sequences in a folder."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List


_FRAME_PATTERN = re.compile(r"(\d+)")


def _frame_number(path: Path) -> int:
    """Extract the last integer in the filename as the frame number."""
    matches = _FRAME_PATTERN.findall(path.stem)
    return int(matches[-1]) if matches else 0


def detect_frame_sequence(folder: Path) -> List[Path]:
    """Return all .exr files in *folder*, sorted by frame number.

    Only files directly in the folder are returned (non-recursive).
    """
    exr_files = sorted(
        [f for f in folder.iterdir() if f.suffix.lower() == ".exr" and f.is_file()],
        key=_frame_number,
    )
    return exr_files
