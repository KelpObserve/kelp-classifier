"""Keep generated documentation links valid when reports use custom output folders."""
import os
from pathlib import Path
import re
from urllib.parse import urlsplit


def relocate_links(text, destination, canonical):
    destination, canonical = Path(destination).resolve(), Path(canonical).resolve()

    def replace(match):
        label, target = match.group(1), match.group(2)
        if urlsplit(target).scheme or target.startswith('#'):
            return match.group(0)
        file, separator, anchor = target.partition('#')
        expected = (canonical / file).resolve()
        # Dataset-local links continue to point to this run's evidence.
        if expected.is_relative_to(canonical):
            return match.group(0)
        # Cross-linked reports may be generated later in the same fresh run.
        # check_repository validates existence after all reports are written.
        relative = Path(os.path.relpath(expected, destination)).as_posix()
        return label + relative + (separator + anchor if separator else '') + ')'

    return re.sub(r'(!?\[[^\]]*\]\()([^)]+)\)', replace, text)
