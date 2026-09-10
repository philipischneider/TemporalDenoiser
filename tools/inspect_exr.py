"""Dev utility: list the layers/channels of one or more EXR files.

Handy for checking what pass names a Blender render actually exported before
configuring the pass mapping in the app.

Usage:
    python tools/inspect_exr.py <folder_or_file> [<folder_or_file> ...]
"""
from __future__ import annotations

import argparse
import os
import sys

import OpenImageIO as oiio


def inspect_file(filepath: str) -> None:
    inp = oiio.ImageInput.open(filepath)
    if not inp:
        print(f"Error opening {filepath}: {oiio.geterror()}")
        return

    print(f"\nFile: {os.path.basename(filepath)}")

    subimage = 0
    while True:
        spec = inp.spec()

        name = spec.getattribute("name")
        if not name:
            name = f"Layer_{subimage}"

        print(f"  Subimage (layer): {name}")
        for ch in spec.channelnames:
            print(f"    Channel: {ch}")

        subimage += 1
        if not inp.seek_subimage(subimage, 0):
            break

    inp.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="EXR file(s) or folder(s) of EXR files")
    args = parser.parse_args()

    exr_files: list[str] = []
    for path in args.paths:
        if os.path.isdir(path):
            exr_files.extend(
                os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".exr")
            )
        elif path.endswith(".exr"):
            exr_files.append(path)

    if not exr_files:
        print("No .exr files found.", file=sys.stderr)
        sys.exit(1)

    for filepath in exr_files:
        inspect_file(filepath)


if __name__ == "__main__":
    main()
