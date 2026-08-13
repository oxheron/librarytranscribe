# librarytranscribe

Transcribe a FLAC library to drum MIDI and fingerprint the original mixes:
recursively scans a directory for .flac files, separates the drum stem with
Demucs, transcribes it with a trained ADT model, writes .mid files into a
mirrored `<root>/midi` tree, records metadata + paths in SQLite, and creates
an audfprint-compatible landmark database.

The model itself is trained in the separate drumtranscribe repo; this repo
only packages the inference tool and hosts the released checkpoints.

## Install and run

```sh
pipx install https://github.com/oxheron/librarytranscribe/releases/download/v0.2.0/librarytranscribe-0.2.0-py3-none-any.whl
librarytranscribe /music/files
```

(`pip install <same url>` works too; pipx just keeps the ~3 GB of
dependencies in an isolated venv. `pip install pipx` if you don't have it.)

On first run the tool downloads the model checkpoint + thresholds.json from
the matching GitHub release into the user cache dir
(`~/.cache/librarytranscribe` on Linux), and Demucs downloads its
`htdemucs_ft` weights (~1 GB) into the torch hub cache. Both are one-time.

## GPU notes

- **NVIDIA on Linux**: works out of the box — the default PyPI torch wheel
  bundles the CUDA runtime, so a working driver is all you need.
- **No GPU**: also works out of the box; the same wheel runs on CPU
  (slowly — expect a few minutes per track).
- **NVIDIA on Windows**: the PyPI torch wheel is CPU-only there. For GPU,
  install the CUDA build first, then the package:

  ```sh
  pip install torch --index-url https://download.pytorch.org/whl/cu126
  pip install <wheel url>
  ```

- **AMD (ROCm, Linux)**: same pattern — install the ROCm torch build first,
  then the package:

  ```sh
  pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
  pip install <wheel url>
  ```

The install-torch-first flows require plain pip in a venv, not pipx: pipx
builds its own isolated venv and would resolve the default (CPU/CUDA) torch
into it, ignoring the one you preinstalled.

Alternative, no install at all — just `librarytranscribe.py` and uv
(dependencies come from its PEP 723 header):

```sh
uv run librarytranscribe.py /music/files
```

The database defaults to `drumlibrary.db` inside the scanned library
directory. Use `-o` (or `--db`) to choose a different name; relative paths
are also placed inside that directory.

The MIDI tree defaults to `<root>/midi`. Use `--midi-dir` to write it
somewhere else (e.g. a local disk when the library lives on a slow network
mount; `-o` and `--audfprint-db` accept absolute paths for the same
reason).

The same run also creates `audfprint.pklz` in the library root. It is updated
incrementally, uses absolute FLAC paths as track identifiers, and uses this
fixed profile:

```text
samplerate: 11025       density: 70        fanout: 8
hashbits: 20            bucketsize: 100    maxtimebits: 17
database shifts: 0      query shifts: 4
```

Use `--audfprint-db NAME.pklz` to choose another path or `--no-audfprint` to
skip fingerprinting. `--force` rebuilds fingerprints as well as MIDI. The
hash table occupies about 404 MiB in memory with this geometry; the `.pklz`
file is gzip-compressed and grows as fingerprints are added. Seventeen time
bits allow roughly 51-minute track offsets and up to 32,767 track IDs.

The output is compatible with audfprint's `match` command. Query with the
requested four sub-frame shifts:

```sh
audfprint match --dbase /music/files/audfprint.pklz --shifts 4 query.wav
```

The `audfprint` match command comes from the upstream audfprint/audfprint2
client; it is not required to create or update the database here.

Query the resulting database:

```sh
sqlite3 /music/files/drumlibrary.db "SELECT artist, album, title, n_onsets FROM tracks"
```

## macOS dependency handling

No separate audfprint package or ffmpeg installation is needed for database
creation: the fingerprinter is packaged with this project and reuses its
existing NumPy/SciPy/librosa stack. This is intentional because the current
`audfprint2` package requires NumPy 2.3+, while Torch 2.2 on Intel macOS
requires NumPy 1.x.

Both the wheel metadata and the inline `uv run` dependency list keep Intel
macOS on the last compatible binary-wheel releases of Torch, torchaudio,
NumPy, Numba, and Demucs. Use Python 3.10–3.12 on Intel Macs. Apple Silicon
uses the normal current-package path and does not need Rosetta.

## Cutting a release

1. Bump `__version__` in `librarytranscribe.py` (this sets both the package
   version and the release-asset URL the tool downloads from).

2. Strip the training checkpoint down to inference weights (the source
   checkpoint comes from the drumtranscribe checkout):

   ```sh
   python strip_checkpoint.py <drumtranscribe>/runs/full_current/best.pt model.pt
   ```

3. Build the wheel (from this directory; output lands in `dist/`):

   ```sh
   uv build
   ```

4. Create a GitHub release on `oxheron/librarytranscribe` tagged
   `v<version>` and upload three assets, named exactly:
   - `model.pt` (the stripped checkpoint)
   - `thresholds.json` (from next to the checkpoint, e.g.
     `../runs/full_current/thresholds.json`)
   - the wheel from `dist/`

   The asset names `model.pt` / `thresholds.json` and the `v<version>` tag
   must match, because the tool's `DEFAULT_MODEL_URL` is built from them.

5. Sanity check in a fresh venv or on a clean machine:

   ```sh
   pipx install <wheel url>
   librarytranscribe ~/somesmalllibrary --limit 1
   ```
