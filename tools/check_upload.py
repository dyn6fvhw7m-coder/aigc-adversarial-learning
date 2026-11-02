"""Check staged Git content without printing any discovered secret value."""

import json
import io
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 1024 * 1024
SECRET = re.compile(
    r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)"
    r"\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"])",
    re.IGNORECASE,
)
FORBIDDEN = {'.pt', '.pth', '.ckpt', '.safetensors', '.pem', '.key', '.pfx', '.p12', '.exe', '.dll'}


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)


def main():
    entries = []
    for record in git('ls-files', '--stage', '-z').split(b'\0'):
        if record:
            metadata, raw_name = record.split(b'\t', 1)
            _, oid, stage = metadata.split()
            if stage != b'0':
                print('BLOCKED: resolve Git conflicts before uploading.')
                return 1
            entries.append((raw_name.decode('utf-8'), oid))
    findings, total = [], 0
    sizes = subprocess.check_output(
        ['git', '-C', str(ROOT), 'cat-file', '--batch-check'],
        input=b''.join(oid + b'\n' for _, oid in entries),
    ).splitlines()
    if len(sizes) != len(entries):
        raise RuntimeError('Unable to inspect staged blob sizes.')
    allowed = []
    for (name, oid), info in zip(entries, sizes):
        path = Path(name)
        if path.suffix.lower() in FORBIDDEN or any(part in {'.venv', '__pycache__', 'backups'} for part in path.parts):
            findings.append((name, 'forbidden runtime/credential file'))
            continue
        _, kind, raw_size = info.split()
        if kind != b'blob':
            findings.append((name, 'unexpected non-file Git entry'))
            continue
        size = int(raw_size)
        total += size
        if size > MAX_BYTES:
            findings.append((name, 'file exceeds 5 MiB review limit'))
            continue
        allowed.append((name, oid))
    if total > 100 * 1024 * 1024:
        findings.append(('<repository>', 'staged content exceeds 100 MiB review limit'))
    if findings:
        for name, reason in findings:
            print(f'BLOCKED: {name}: {reason}')
        return 1
    blobs = io.BytesIO(subprocess.check_output(
        ['git', '-C', str(ROOT), 'cat-file', '--batch'],
        input=b''.join(oid + b'\n' for _, oid in allowed),
    ))
    for name, oid in allowed:
        path = Path(name)
        actual_oid, kind, raw_size = blobs.readline().split()
        if actual_oid != oid or kind != b'blob':
            raise RuntimeError('Unexpected staged blob response.')
        data = blobs.read(int(raw_size))
        if blobs.read(1) != b'\n':
            raise RuntimeError('Incomplete staged blob.')
        try:
            text = data.decode('utf-8-sig')
        except UnicodeDecodeError:
            continue
        candidates = [text]
        if path.suffix == '.ipynb':
            candidates.extend(strings(json.loads(text)))
        if any(SECRET.search(candidate) for candidate in candidates):
            findings.append((name, 'potential embedded credential; value withheld'))
    if findings:
        for name, reason in findings:
            print(f'BLOCKED: {name}: {reason}')
        return 1
    print(f'PASS: {len(entries)} staged files; {total / 1024 / 1024:.2f} MiB; no blocked content found.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
