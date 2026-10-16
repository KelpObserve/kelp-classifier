"""Regression checks for policy-preserving acquisition and portable evidence checks."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_pipeline import ROOT, load_module

candidates = load_module('candidates', ROOT / 'data_collection/prepare_acquisition_candidates.py')
repository = load_module('repository', ROOT / 'scripts/check_repository.py')
artifacts = load_module('artifacts', ROOT / 'scripts/check_artifacts.py')


class RepositoryToolsTests(unittest.TestCase):
    def test_acquisition_preserves_current_policy_and_all_columns(self):
        source = ROOT / 'data_collection/s2_sesimbra_like_observations.csv'
        with source.open() as stream:
            original = list(csv.DictReader(stream))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'candidates'
            result = candidates.prepare(source, output)
            self.assertEqual(result['counts'], {'PRESENT': 8275, 'ABSENT': 219})
            self.assertEqual(result['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
            for label in ['PRESENT', 'ABSENT']:
                with (output / f'{label.lower()}.csv').open() as stream:
                    self.assertEqual(list(csv.DictReader(stream)),
                                     [r for r in original if r['occurrenceStatus'] == label])
            with self.assertRaises(FileExistsError):
                candidates.prepare(source, output)
            with self.assertRaises(ValueError):
                candidates.prepare(ROOT / 'data_collection/deduped_absent_candidates.csv', Path(directory) / 'bad')
            self.assertFalse((Path(directory) / 'bad').exists())

    def test_checker_covers_non_markdown_urls_and_private_references(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as directory:
            root = Path(directory)
            source = root / 'sample.json'
            source.write_text(json.dumps({'url': ('https:' + '//example.org/asset.tif'),
                                          'private': '/' + 'home' + '/example/input'}))
            vcs = root / 'requirements.lock'
            vcs.write_text(('package @ git+https:' + '//github.com/example/model.git@') + 'a'*40)
            with patch.object(repository, 'files', return_value=[source, vcs]):
                problems, urls, _ = repository.check()
            self.assertTrue(any('Machine-specific' in problem for problem in problems))
            self.assertIn(('https:' + '//example.org/asset.tif'), urls)
            self.assertIn(('https:' + '//github.com/example/model/commit/') + 'a'*40, urls)

    def test_artifact_hash_mismatch_missing_file_and_group_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'artifact'
            path.write_bytes(b'published bytes')
            manifest = {'artifacts': [{'id': 'example', 'path_key': 'CLAY_CKPT',
                                      'group': 'training', 'sha256': artifacts.digest(path)}]}
            with patch.object(artifacts.config, 'get_path', return_value=path):
                self.assertEqual(artifacts.verify(manifest, None), [])
                path.write_bytes(b'changed bytes')
                self.assertIn('checksum mismatch', artifacts.verify(manifest, None)[0])
                self.assertEqual(artifacts.verify(manifest, ['field']), [])
                path.unlink()
                self.assertIn('missing', artifacts.verify(manifest, None)[0])


if __name__ == '__main__':
    unittest.main()
