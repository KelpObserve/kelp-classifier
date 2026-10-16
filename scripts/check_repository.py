"""Check active repository syntax, configuration, documentation and optional web links."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {'.py', '.md', '.json', '.yaml', '.yml', '.html', '.csv', '.txt', '.lock'}
URL = re.compile(r"https?://[^\s<>\"'\)]+")
MACHINE_REFERENCE = re.compile(
    r'\b(?:server|laptop)\b|SERVER_|/(?:Users|home|mnt|Volumes|private/tmp)/|'+
    r'(?:ssh|sftp|file)://|\blocalhost\b|\b127\.0\.0\.1\b|[A-Za-z]:\\|'
    r'configured file \(see README\)', re.I)
LINK = re.compile(r'!?\[[^\]]*\]\(([^)]+)\)')


def files():
    result = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT, capture_output=True)
    if result.returncode == 0:
        names = result.stdout.decode().split('\0')
        return sorted({ROOT / name for name in names if name and (ROOT / name).is_file()})
    # Source archives are also supported; no Git metadata is required to run checks.
    excluded = {'.git', '.local_archive', '__pycache__', '.venv', 'runs', 'data', 'backups', 'logs', 'tmp'}
    return sorted(p for p in ROOT.rglob('*') if p.is_file()
                  and not excluded.intersection(p.relative_to(ROOT).parts)
                  and p.name != 'config.json' and p.suffix not in {'.pyc', '.npz', '.h5', '.pt', '.joblib'})



def anchors(path):
    counts, result = {}, set()
    for title in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', path.read_text(), re.M):
        slug = re.sub(r'[^\w\- ]', '', title.lower()).replace(' ', '-')
        count = counts.get(slug, 0)
        result.add(slug if count == 0 else f'{slug}-{count}')
        counts[slug] = count + 1
    return result


def probe(url):
    try:
        request = Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
        try:
            response = urlopen(request, timeout=15)
        except HTTPError as exc:
            if exc.code != 405:
                raise
            response = urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Range': 'bytes=0-0'}), timeout=15)
        with response:
            return ('OK', response.status, url)
    except HTTPError as exc:
        return ('BROKEN' if exc.code in {404, 410} else 'UNVERIFIED', exc.code, url)
    except (URLError, TimeoutError, OSError) as exc:
        return ('UNVERIFIED', str(exc), url)


def check():
    problems, urls = [], set()
    active = files()
    defaults = json.loads((ROOT / 'example_config.json').read_text())['paths']
    for key, value in defaults.items():
        for ref in re.findall(r'\{([^}]+)\}', value):
            if ref not in defaults and ref != 'REPO_ROOT':
                problems.append(f'Unknown configuration reference {key}: {ref}')
    resources = json.loads((ROOT / 'example_config.json').read_text()).get('resources', {})
    for key, resource in resources.items():
        for field in ['reference', 'producer', 'protocol']:
            value = resource.get(field)
            if value and not urlsplit(value).scheme and not (ROOT / value).exists():
                problems.append(f'Missing resource {key}.{field}: {value}')
    inventory = json.loads((ROOT / 'configs/artifacts.json').read_text())
    seen = set()
    for item in inventory['artifacts']:
        if item['id'] in seen:
            problems.append(f"Duplicate artifact ID: {item['id']}")
        seen.add(item['id'])
        if item['path_key'] not in defaults or not re.fullmatch(r'[0-9a-f]{64}', item['sha256']):
            problems.append(f"Invalid artifact key or checksum: {item['id']}")
        child = Path(item.get('relative_path', ''))
        if child.is_absolute() or '..' in child.parts:
            problems.append(f"Artifact child must stay within its configured directory: {item['id']}")
        if item.get('catalog') and not (ROOT / item['catalog']).is_file():
            problems.append(f"Missing artifact catalog: {item['id']}")
    for path in active:
        if path.is_symlink():
            problems.append(f'Symbolic link: {path.relative_to(ROOT)}')
        if path.suffix not in TEXT_SUFFIXES:
            continue
        text = path.read_text()
        rel = path.relative_to(ROOT)
        if path.name != 'check_repository.py' and MACHINE_REFERENCE.search(text):
            problems.append(f'Machine-specific wording or placeholder: {rel}')
        for raw in URL.findall(text) if path.name != 'check_repository.py' else []:
            url = raw.rstrip('.,;')
            # pip VCS requirements name a revision, not a literal HTTP URL.
            url = re.sub(r'^(https://github\.com/[^/]+/[^/]+)\.git@([0-9a-f]{40})$',
                         r'\1/commit/\2', url)
            urls.add(url)
        if path.suffix == '.py':
            try:
                ast.parse(text)
            except SyntaxError as exc:
                problems.append(f'{rel}: {exc}')
            for key in re.findall(r'get_path\([\"\']([A-Z][A-Z_]+)[\"\']\)', text):
                if key not in defaults:
                    problems.append(f'{rel}: unknown configuration key {key}')
        if path.suffix != '.md':
            continue
        for match in LINK.finditer(text):
            target = match.group(1).split(' "')[0]
            if target.startswith(('http://', 'https://')):
                urls.add(target)
                continue
            if urlsplit(target).scheme:
                continue
            name, _, anchor = target.partition('#')
            destination = path.parent / unquote(name) if name else path
            if not destination.exists():
                problems.append(f'{rel}: missing link target {target}')
            elif destination.is_file() and anchor and destination.suffix == '.md' and unquote(anchor) not in anchors(destination):
                problems.append(f'{rel}: missing anchor {target}')
        for script in re.findall(r'\bpython(?:3(?:\.11)?)?\s+([\w./-]+\.py)', text):
            if not (ROOT / script).is_file():
                problems.append(f'{rel}: missing command script {script}')
    return problems, urls, active


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external', action='store_true')
    args = parser.parse_args()
    problems, urls, active = check()
    if args.external:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for state, detail, url in pool.map(probe, sorted(urls)):
                print(state, detail, url)
                if state == 'BROKEN':
                    problems.append(f'Broken external link: {url}')
    for problem in problems:
        print('ERROR:', problem)
    print(f'Checked {len(active)} active files; {len(problems)} errors; {len(urls)} external links.')
    raise SystemExit(bool(problems))
