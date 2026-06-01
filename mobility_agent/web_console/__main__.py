from __future__ import annotations

import argparse
import os

import uvicorn

from .api import create_app
from .config import WebConsoleSettings


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the script_new web console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--job-root", action="append", dest="job_roots", default=[])
    parser.add_argument("--repo-root", default=os.getcwd())
    args = parser.parse_args()

    settings = WebConsoleSettings.from_repo(
        args.repo_root,
        job_roots=args.job_roots or [args.repo_root],
        host=args.host,
        port=args.port,
    )
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
