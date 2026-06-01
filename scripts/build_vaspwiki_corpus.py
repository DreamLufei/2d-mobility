#!/usr/bin/env python3
from __future__ import annotations


def main() -> int:
    raise RuntimeError(
        "scripts/build_vaspwiki_corpus.py is deprecated. "
        "Use scripts/sync_vasp_wiki.py and scripts/reindex_vasp_wiki.py with MOBILITY_DB_URI instead."
    )


if __name__ == "__main__":
    raise SystemExit(main())
