"""Download and extract the configured JNCC v20220124 public Access snapshot."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    target = config.get_path('JNCC_DATABASE')
    if target.exists():
        raise FileExistsError('Configured database exists; refusing to overwrite it')
    target.parent.mkdir(parents=True, exist_ok=True)
    url = config.get_service('jncc_snapshot')
    with tempfile.TemporaryDirectory(dir=target.parent) as directory:
        archive = Path(directory) / 'snapshot.zip'
        with urlopen(url, timeout=90) as response, archive.open('wb') as output:
            shutil.copyfileobj(response, output)
        with ZipFile(archive) as bundle:
            members = [m for m in bundle.infolist()
                       if Path(m.filename).name == 'SnapshotDatav52_PUBLIC_20220124.mdb']
            if len(members) != 1:
                raise ValueError(f'Expected the primary v20220124 database, found {len(members)}')
            extracted = Path(directory) / 'database.mdb'
            with bundle.open(members[0]) as source, extracted.open('wb') as output:
                shutil.copyfileobj(source, output)
        provenance = dict(source=url, snapshot='v20220124', member=members[0].filename,
                          archive_sha256=sha256(archive), database_sha256=sha256(extracted))
        extracted.rename(target)
        target.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
