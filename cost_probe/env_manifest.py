#!/usr/bin/env python3
"""Digest the running interpreter's environment for exact cross-machine comparison.

    <venv>/bin/python cost_probe/env_manifest.py \
        --overlay runtime=DIR --overlay runtime_sim=DIR --out manifest.json

    python cost_probe/env_manifest.py --compare A.json B.json

Covers: the base python installation (per top-level entry of sys.base_prefix),
every distribution in the venv site-packages and each --overlay dir (content
digests split into package vs dist-info metadata), and unowned on-disk files.
--compare exits 1 on any difference other than dist-info metadata digests.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
import sysconfig
from pathlib import Path

SKIP_RECORD_PREFIX = ('../', 'bin/')
SKIP_META_FILES = {'RECORD', 'INSTALLER', 'REQUESTED', 'direct_url.json', 'uv_cache.json'}


def norm_name(s):
    return re.sub(r'[-_.]+', '-', s).lower()


def file_digest(p):
    """(sha256 hex, size) of a path; symlinks hash the link target, dirs a sentinel."""
    if os.path.islink(p):
        data = os.readlink(p).encode()
        return hashlib.sha256(data).hexdigest(), len(data)
    if os.path.isdir(p):
        return hashlib.sha256(b'<DIR>').hexdigest(), 0
    h = hashlib.sha256()
    size = 0
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def walk_files(root, relbase):
    """relpath -> (sha256, size) for every file under root, skipping
    __pycache__ dirs and *.pyc. Symlinked dirs record their link target."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        keep = []
        for d in sorted(dirnames):
            p = os.path.join(dirpath, d)
            if d == '__pycache__':
                continue
            if os.path.islink(p):
                out[os.path.relpath(p, relbase)] = file_digest(p)
            else:
                keep.append(d)
        dirnames[:] = keep
        for f in sorted(filenames):
            if f.endswith('.pyc') or '__pycache__' in f:
                continue
            p = os.path.join(dirpath, f)
            out[os.path.relpath(p, relbase)] = file_digest(p)
    return out


def tree_digest(entries):
    """entries: relpath -> (hex, size). Digest of the sorted 'relpath hex' lines."""
    return hashlib.sha256(
        ''.join(f'{rel} {hex_}\n' for rel, (hex_, _) in sorted(entries.items())).encode()
    ).hexdigest()


def record_entries(di):
    rec = di / 'RECORD'
    if not rec.exists():
        return []
    with open(rec, newline='') as f:
        return [row[0][2:] if row[0].startswith('./') else row[0]
                for row in csv.reader(f) if row]


def dist_name_version(di):
    name = version = None
    meta = di / 'METADATA'
    if meta.exists():
        for line in meta.read_text(errors='replace').splitlines():
            if line.startswith('Name:'):
                name = line.split(':', 1)[1].strip()
            elif line.startswith('Version:'):
                version = line.split(':', 1)[1].strip()
            if name and version:
                break
    if not (name and version):  # fall back to the dir name
        name, _, version = di.name[:-len('.dist-info')].rpartition('-')
    return name, version


def site_section(root):
    root = Path(root)
    dists, owned = {}, set()
    for di in sorted(root.glob('*.dist-info')):
        name, version = dist_name_version(di)
        dprefix = di.name + '/'
        entries = record_entries(di)
        owned.update(e for e in entries if not e.startswith('../') and not e.startswith('/'))
        pkg, meta_files, missing = {}, {}, []
        n_bytes = 0
        for e in entries:
            if e.startswith(SKIP_RECORD_PREFIX) or e.startswith('/'):
                continue
            if '__pycache__/' in e or e.endswith('.pyc'):
                continue
            if e.startswith(dprefix) and e[len(dprefix):] in SKIP_META_FILES:
                continue
            p = root / e
            if not os.path.lexists(p):
                missing.append(e)
                continue
            hex_, size = file_digest(p)
            n_bytes += size
            (meta_files if e.startswith(dprefix) else pkg)[e] = (hex_, size)
        dists[norm_name(name)] = {
            'name': name, 'version': version, 'dist_info': di.name,
            'package_digest': tree_digest(pkg), 'metadata_digest': tree_digest(meta_files),
            'n_files': len(pkg) + len(meta_files), 'n_bytes': n_bytes,
            'missing': missing,
        }
    unowned = {}
    for rel, (hex_, size) in walk_files(root, root).items():
        top = rel.split('/')[0]
        if top == 'bin' or (top.startswith('.') and '/' not in rel):
            continue
        if rel not in owned:
            unowned[rel] = hex_
    return {'root': str(root), 'dists': dists, 'unowned': unowned}


def python_section():
    bp = Path(sys.base_prefix)
    build = (bp / 'BUILD').read_text().strip() if (bp / 'BUILD').exists() else None
    install = {}
    for e in sorted(os.listdir(bp)):
        p = bp / e
        if p.is_dir() and not os.path.islink(p):
            install[e] = tree_digest(walk_files(p, bp))
        else:
            install[e] = file_digest(p)[0]
    return {
        'version': platform.python_version(),
        'version_full': sys.version,
        'build': build,
        'base_prefix': str(bp),
        'install': install,
    }


def build_manifest(overlays):
    return {
        'tool': 'env_manifest', 'format': 1,
        'python': python_section(),
        'sites': {'site': site_section(sysconfig.get_paths()['purelib'])}
                 | {name: site_section(d) for name, d in overlays},
    }


def short(v):
    return None if v is None else str(v)[:16]


def compare(a, b):
    hard, meta_only = [], []
    pa, pb = a.get('python', {}), b.get('python', {})
    for k in ('version', 'version_full', 'build'):
        if pa.get(k) != pb.get(k):
            hard.append(f'python.{k}: {str(pa.get(k))[:60]!r} != {str(pb.get(k))[:60]!r}')
    ia, ib = pa.get('install', {}), pb.get('install', {})
    for e in sorted(set(ia) | set(ib)):
        if ia.get(e) != ib.get(e):
            hard.append(f'python.install[{e}]: {short(ia.get(e))} != {short(ib.get(e))}')
    sa_all, sb_all = a.get('sites', {}), b.get('sites', {})
    for site in sorted(set(sa_all) | set(sb_all)):
        sa, sb = sa_all.get(site), sb_all.get(site)
        if sa is None or sb is None:
            hard.append(f'{site}: site present in only one manifest')
            continue
        da, db = sa['dists'], sb['dists']
        for k in sorted(set(da) | set(db)):
            xa, xb = da.get(k), db.get(k)
            tag = f'{site}:{k}'
            if xa is None:
                hard.append(f'{tag}: extra in B ({xb["name"]}=={xb["version"]})')
                continue
            if xb is None:
                hard.append(f'{tag}: missing in B ({xa["name"]}=={xa["version"]})')
                continue
            if xa['version'] != xb['version']:
                hard.append(f'{tag}: version {xa["version"]} != {xb["version"]}')
            if xa['package_digest'] != xb['package_digest']:
                hard.append(f'{tag}: package_digest {short(xa["package_digest"])} != {short(xb["package_digest"])}')
            if xa['metadata_digest'] != xb['metadata_digest']:
                meta_only.append(f'{tag}: metadata_digest {short(xa["metadata_digest"])} != {short(xb["metadata_digest"])}')
            if xa['missing'] != xb['missing']:
                hard.append(f'{tag}: missing differ A={xa["missing"][:3]} B={xb["missing"][:3]}')
        ua, ub = sa['unowned'], sb['unowned']
        for k in sorted(set(ua) | set(ub)):
            if k not in ua:
                hard.append(f'{site}: unowned {k} only in B')
            elif k not in ub:
                hard.append(f'{site}: unowned {k} only in A')
            elif ua[k] != ub[k]:
                hard.append(f'{site}: unowned {k} differs')
    return hard, meta_only


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--overlay', action='append', default=[],
                    help='NAME=DIR extra package dir to manifest (repeatable)')
    ap.add_argument('--out', help='write the manifest JSON here')
    ap.add_argument('--compare', nargs=2, metavar=('A.json', 'B.json'),
                    help='compare two manifests; exit 1 on non-metadata differences')
    args = ap.parse_args()
    if args.compare:
        a = json.loads(Path(args.compare[0]).read_text())
        b = json.loads(Path(args.compare[1]).read_text())
        hard, meta_only = compare(a, b)
        for line in meta_only:
            print('META-DIFF ' + line)
        for line in hard:
            print('DIFF ' + line)
        if not hard and not meta_only:
            print('IDENTICAL')
        elif not hard:
            print('IDENTICAL except metadata digests')
        return 1 if hard else 0
    overlays = []
    for spec in args.overlay:
        name, sep, d = spec.partition('=')
        assert sep and name != 'site', f'bad --overlay {spec!r}'
        overlays.append((name, d))
    manifest = build_manifest(overlays)
    text = json.dumps(manifest, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + '\n')
        print('wrote', args.out)
    else:
        print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
