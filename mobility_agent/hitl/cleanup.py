from __future__ import annotations

import os
import shutil

from ..graph.stage_contracts import CleanupPolicyName, CleanupPreview, build_cleanup_preview


def preview_cleanup(*, workdir: str, resume_stage: str, cleanup_policy: CleanupPolicyName) -> CleanupPreview:
    return build_cleanup_preview(workdir=workdir, resume_stage=resume_stage, cleanup_policy=cleanup_policy)


def apply_cleanup(*, workdir: str, preview: CleanupPreview) -> CleanupPreview:
    for path in preview.invalidated_artifacts:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            continue
        try:
            os.remove(path)
        except FileNotFoundError:
            continue
        except IsADirectoryError:
            shutil.rmtree(path, ignore_errors=True)
    return preview
