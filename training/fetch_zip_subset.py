"""Extract only some files from a huge remote .zip, using HTTP Range requests. Resumable.

    python training/fetch_zip_subset.py URL OUT_DIR FILTER [FILTER ...]

Downloads the zip's table of contents (at the end of the file), then only the byte ranges that
hold entries whose path contains one of the FILTER strings, in 8 MB chunks with retries.
Bytes go into a sparse local copy (OUT_DIR/.partial.zip) at their original offsets, so a
normal zipfile.ZipFile can read the entries. Used because slow/flaky connections kept failing
on 1-2 GB downloads, and we need only part of each archive.
"""
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

CHUNK = 8 * 1024 * 1024


def fetch(url, start, end, tries=8):
    """Bytes [start, end] inclusive."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "leafscan-fetch"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == end - start + 1:
                return data
        except Exception as exc:  # network hiccup: wait and retry
            print(f"  retry {attempt + 1} ({exc.__class__.__name__})", flush=True)
        time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"could not fetch bytes {start}-{end}")


def size_of(url):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "leafscan-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Range"].split("/")[1])


def main(url, out_dir, filters):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    part, state_file = out / ".partial.zip", out / ".fetch_state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"done": []}
    total = state.get("size") or size_of(url)
    state["size"] = total
    if not part.exists():
        with open(part, "wb") as f:
            f.truncate(total)  # sparse file of the full size

    def write(start, data):
        with open(part, "r+b") as f:
            f.seek(start)
            f.write(data)

    if "toc" not in state:  # table of contents lives in the last few MB
        tail = min(total, 40 * 1024 * 1024)
        print(f"fetching table of contents ({tail / 1e6:.0f} MB)", flush=True)
        for s in range(total - tail, total, CHUNK):
            write(s, fetch(url, s, min(s + CHUNK, total) - 1))
        state["toc"] = True
        state_file.write_text(json.dumps(state))

    zf = zipfile.ZipFile(part)
    wanted = [i for i in zf.infolist() if not i.is_dir() and any(f in i.filename for f in filters)]
    if not wanted:
        raise SystemExit("no entries match " + repr(filters))
    # Which 8 MB blocks (aligned to multiples of CHUNK) hold the wanted entries?
    blocks = set()
    for i in wanted:
        end = min(total - 1, i.header_offset + 30 + len(i.filename.encode()) + i.compress_size + 1024)
        blocks.update(range(i.header_offset // CHUNK, end // CHUNK + 1))
    chunks = sorted(b * CHUNK for b in blocks)
    print(f"{len(wanted)} entries in {len(chunks)} blocks ({len(chunks) * CHUNK / 1e6:.0f} MB)", flush=True)

    done = set(state["done"])
    for n, s in enumerate(chunks, 1):
        if s in done:
            continue
        write(s, fetch(url, s, min(s + CHUNK, total) - 1))
        done.add(s)
        state["done"] = sorted(done)
        state_file.write_text(json.dumps(state))
        print(f"  {n}/{len(chunks)} chunks", flush=True)

    zf.close()
    root = out.resolve()
    with zipfile.ZipFile(part) as zf:
        for info in wanted:
            target = (out / PurePosixPath(info.filename)).resolve()
            if root not in target.parents:  # never write outside OUT_DIR
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
    part.unlink()
    print(f"extracted {len(wanted)} files to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
