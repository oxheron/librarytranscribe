#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mutagen",
#     "torch>=2.0,<2.3; sys_platform == 'darwin' and platform_machine == 'x86_64'",
#     "torch>=2.0; sys_platform != 'darwin' or platform_machine != 'x86_64'",
#     "numpy<2; sys_platform == 'darwin' and platform_machine == 'x86_64'",
#     "numpy; sys_platform != 'darwin' or platform_machine != 'x86_64'",
#     "numba<0.63; sys_platform == 'darwin' and platform_machine == 'x86_64'",
#     "scipy",
#     "librosa",
#     "soundfile",
#     "pretty_midi",
#     "demucs>=4.1; sys_platform != 'darwin' or platform_machine != 'x86_64'",
#     "demucs>=4.0.1,<4.1; sys_platform == 'darwin' and platform_machine == 'x86_64'",
#     "torchaudio<2.3; sys_platform == 'darwin' and platform_machine == 'x86_64'",
#     "conformer",
# ]
# ///
# The environment markers keep Intel macs (the one platform recent torch,
# numba/llvmlite and demucs/sphn wheels dropped) on the last releases that
# still ship prebuilt wheels, so nothing is ever compiled from source; all
# other platforms get the latest versions (see pyproject.toml for details).
# On Intel macs use Python <= 3.12 (the last torch 2.2.2 wheel is cp312).
"""
Build a drum-transcription database from a FLAC library.

Recursively finds .flac files under a root directory, reads their metadata,
transcribes each one to drum MIDI with a trained ADT model (+ calibrated
per-class thresholds), writes the .mid files into a "midi" subdirectory of
the root that mirrors the library layout, and records metadata + both file
paths in a SQLite database.

  /music/files/Artist/Album/01 - Song.flac
      -> /music/files/midi/Artist/Album/01 - Song.mid

Works on Linux, macOS and Windows (pure pathlib path handling, "~" is
expanded, .FLAC/.flac both matched).

This file is fully self-contained — it does NOT import anything from the
drumtranscribe repo. Three ways to run it elsewhere:

  1. As an installed package (see README.md):
       pipx install <librarytranscribe wheel>
       librarytranscribe /music/files
  2. With uv — dependencies come from the PEP 723 header above:
       uv run librarytranscribe.py /music/files
  3. As a bare script:
       pip install mutagen torch numpy scipy librosa soundfile pretty_midi \
                   demucs conformer
       python librarytranscribe.py /music/files

When --model is not given, the release checkpoint and its thresholds.json
are downloaded once into the user cache directory and reused from there.
Pass --model best.pt to use a local checkpoint instead (calibrate.py writes
its thresholds.json next to it).

Usage:
  librarytranscribe /music/files --db drums.db [--model runs/full/best.pt]

Metadata rules:
  - artist, title and album are REQUIRED to save a track; date, genre,
    track/disc numbers and totals are optional.
  - Fields missing from the FLAC tags are filled from the directory layout,
    assuming  Artist/Album/NN - Title.flac  (NN = track number; the "NN"
    prefix and separator are optional when parsing the title). Existing tags
    are never overwritten.

Reruns are incremental: files already transcribed with unchanged size/mtime
are skipped (use --force to redo them). Failures are logged into the DB and
retried on the next run.

Query examples:
  sqlite3 drums.db "SELECT artist, album, title, n_onsets FROM tracks"
  sqlite3 drums.db "SELECT flac_path, midi_path FROM tracks WHERE artist='X'"
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

__version__ = "0.1.0"

# Where the default checkpoint is fetched from when --model is not given.
# Publishing a new model means uploading model.pt + thresholds.json to a
# GitHub release tagged v<__version__> (see README.md).
_RELEASE_URL = ("https://github.com/oxheron/librarytranscribe/releases/download/"
                f"v{__version__}")
DEFAULT_MODEL_URL = f"{_RELEASE_URL}/model.pt"
DEFAULT_THRESHOLDS_URL = f"{_RELEASE_URL}/thresholds.json"


# =========================================================================
# Transcription constants (mirrors drumtranscribe src/: features.py, peak.py,
# data/class_map.py — keep in sync with the checkpoint's training code)
# =========================================================================

SR = 44100
N_MELS = 256
HOP_LENGTH = 441
N_FFT = 2048
HOP_SEC = HOP_LENGTH / SR

# The drums-fine-tuned member of the htdemucs_ft bag. The bag's weight
# matrix is diagonal, so its drums stem comes entirely from this one model —
# loading it alone gives identical drums output ~4x faster than the full bag.
DEMUCS_MODEL = "f7e0c4bc"

# (name, output GM pitch) — index order is the model's class order.
CLASS_DEFS = [
    ("kick", 36), ("snare", 38), ("sidestick", 37),
    ("hihat_closed", 42), ("hihat_open", 46),
    ("tom_low", 45), ("tom_mid", 47), ("tom_high", 50),
    ("crash", 49), ("ride", 51), ("perc", 56),
]
NUM_CLASSES = len(CLASS_DEFS)
CLASS_NAMES = [d[0] for d in CLASS_DEFS]
GM_PITCHES = [d[1] for d in CLASS_DEFS]
NAME_TO_CLASS = {n: i for i, (n, _) in enumerate(CLASS_DEFS)}

# Per-class peak-picking refractory windows (ms).
DEFAULT_MIN_DIST_MS = {
    "kick": 20.0, "snare": 12.0, "sidestick": 20.0,
    "hihat_closed": 30.0, "hihat_open": 40.0,
    "tom_low": 15.0, "tom_mid": 15.0, "tom_high": 15.0,
    "crash": 50.0, "ride": 30.0, "perc": 30.0,
}

# Timekeeping surfaces reconciled by stream-consistency voting.
_GROUP_SURFACE = {
    NAME_TO_CLASS["hihat_closed"]: "hat",
    NAME_TO_CLASS["hihat_open"]: "hat",
    NAME_TO_CLASS["ride"]: "ride",
}


# =========================================================================
# Model (vendored from src/model.py; needs `pip install conformer torch`)
# =========================================================================

def _build_model_classes():
    import torch
    import torch.nn as nn
    from conformer import ConformerBlock

    class UnfusedLayerNorm(nn.Module):
        """LayerNorm from primitive ops; reuses the original weight/bias
        Parameters so state_dict keys stay identical to nn.LayerNorm."""

        def __init__(self, layernorm: nn.LayerNorm):
            super().__init__()
            self.normalized_shape = tuple(layernorm.normalized_shape)
            self.eps = layernorm.eps
            self.weight = layernorm.weight
            self.bias = layernorm.bias

        def forward(self, x):
            dims = tuple(range(-len(self.normalized_shape), 0))
            orig_dtype = x.dtype
            xf = x.float()
            mean = xf.mean(dim=dims, keepdim=True)
            var = xf.var(dim=dims, keepdim=True, correction=0)
            xhat = (xf - mean) * torch.rsqrt(var + self.eps)
            if self.weight is not None:
                xhat = xhat * self.weight.float() + self.bias.float()
            return xhat.to(orig_dtype)

    def replace_layernorms(module: nn.Module) -> nn.Module:
        for name, child in module.named_children():
            if isinstance(child, nn.LayerNorm):
                setattr(module, name, UnfusedLayerNorm(child))
            else:
                replace_layernorms(child)
        return module

    class CNNFrontend(nn.Module):
        def __init__(self, d_model=256):
            super().__init__()

            def block(in_ch, out_ch):
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(8, out_ch),
                    nn.GELU(),
                )

            self.stem = nn.Sequential(
                block(1, 32), nn.MaxPool2d((2, 1)),
                block(32, 64), nn.MaxPool2d((2, 1)),
                block(64, 128), nn.MaxPool2d((2, 1)),
            )
            self.res = block(128, 128)

        def forward(self, x):
            return self.res(self.stem(x))

    class Tokenizer(nn.Module):
        def __init__(self, in_features, d_model=256):
            super().__init__()
            self.proj = nn.Linear(in_features, d_model)

        def forward(self, x):
            b, c, f, t = x.shape
            x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
            return self.proj(x)

    class ConformerEncoder(nn.Module):
        def __init__(self, d_model=256, n_layers=8, n_heads=4, ff_mult=4, dropout=0.1):
            super().__init__()
            self.layers = nn.ModuleList([
                ConformerBlock(
                    dim=d_model,
                    dim_head=d_model // n_heads,
                    heads=n_heads,
                    ff_mult=ff_mult,
                    conv_expansion_factor=2,
                    conv_kernel_size=15,
                    attn_dropout=dropout,
                    ff_dropout=dropout,
                    conv_dropout=dropout,
                ) for _ in range(n_layers)
            ])
            replace_layernorms(self)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class ADTModel(nn.Module):
        def __init__(self, n_classes=NUM_CLASSES, d_model=256, n_layers=8, n_mels=N_MELS):
            super().__init__()
            f_prime = n_mels // 8
            self.cnn = CNNFrontend(d_model)
            self.tokenizer = Tokenizer(128 * f_prime, d_model)
            self.encoder = ConformerEncoder(d_model, n_layers)
            self.head = nn.Linear(d_model, n_classes)

        def forward(self, x):
            x = self.cnn(x)
            x = self.tokenizer(x)
            x = self.encoder(x)
            return self.head(x)

    return ADTModel


def load_model_from_checkpoint(path: str, device):
    import torch

    ADTModel = _build_model_classes()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_cfg = ckpt.get("config", {}).get("model", {})
    model = ADTModel(
        n_classes=NUM_CLASSES,
        d_model=model_cfg.get("d_model", 256),
        n_layers=model_cfg.get("n_layers", 8),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# =========================================================================
# Features + windowed inference (vendored from src/features.py, src/eval.py)
# =========================================================================

def extract_logmel_array(y, sr: int = SR):
    import librosa
    import numpy as np

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=20, fmax=20000,
    )
    mel = np.maximum(mel, 1e-10)
    return librosa.power_to_db(mel, ref=1.0, top_db=80.0).astype(np.float32)


def predict_track(model, audio, device, seg_frames=400, batch_size=8, use_bf16=True):
    """Model over the full track in 50%-overlapping windows; stitched sigmoid
    probabilities, averaged over overlaps. Returns [T, n_classes] float32."""
    import numpy as np
    import torch

    mel = extract_logmel_array(audio)
    n_frames = mel.shape[1]
    hop = seg_frames // 2

    starts = list(range(0, max(n_frames - seg_frames, 0) + 1, hop))
    if not starts:
        starts = [0]
    if starts[-1] + seg_frames < n_frames:
        starts.append(n_frames - seg_frames)

    pad_to = max(n_frames, seg_frames)
    mel_padded = np.pad(mel, ((0, 0), (0, pad_to - n_frames)), constant_values=mel.min())

    probs_sum = np.zeros((pad_to, NUM_CLASSES), dtype=np.float64)
    counts = np.zeros(pad_to, dtype=np.float64)

    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            chunk_starts = starts[i: i + batch_size]
            batch = np.stack([mel_padded[:, s: s + seg_frames] for s in chunk_starts])
            x = torch.from_numpy(batch).unsqueeze(1).to(device)
            if use_bf16 and device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
            else:
                logits = model(x)
            p = torch.sigmoid(logits.float()).cpu().numpy()
            for s, pj in zip(chunk_starts, p):
                probs_sum[s: s + seg_frames] += pj
                counts[s: s + seg_frames] += 1.0

    counts = np.maximum(counts, 1.0)
    return (probs_sum / counts[:, None])[:n_frames].astype(np.float32)


# =========================================================================
# Onset decoding + MIDI (vendored from src/peak.py)
# =========================================================================

def _refine_onset(col, f: int, hop_sec: float) -> float:
    import numpy as np

    T = len(col)
    if f <= 0 or f >= T - 1:
        delta = 0.0
    else:
        ym1, y0, yp1 = float(col[f - 1]), float(col[f]), float(col[f + 1])
        denom = ym1 - 2.0 * y0 + yp1
        if abs(denom) < 1e-12:
            delta = 0.0
        else:
            delta = 0.5 * (ym1 - yp1) / denom
            if abs(delta) > 0.5:
                delta = float(np.clip(delta, -0.5, 0.5))
    return max((f + delta) * hop_sec, 0.0)


def decode_onsets(probs, thresholds=0.5, hop_sec=HOP_SEC):
    """probs [T, n_classes] -> [(onset_sec, class_idx, confidence)], sorted.
    Per-class refractory windows from DEFAULT_MIN_DIST_MS; parabolic
    sub-frame timing refinement."""
    import numpy as np
    from scipy.signal import find_peaks

    n_classes = probs.shape[1]
    thr = np.broadcast_to(np.asarray(thresholds, dtype=np.float64), (n_classes,))
    min_dist_frames = np.maximum(
        (np.array([DEFAULT_MIN_DIST_MS[n] for n in CLASS_NAMES]) / 1000.0 / hop_sec)
        .astype(int), 1,
    )

    events = []
    for c in range(n_classes):
        col = probs[:, c]
        peaks, props = find_peaks(col, height=thr[c], distance=int(min_dist_frames[c]))
        for frame, height in zip(peaks, props["peak_heights"]):
            events.append((_refine_onset(col, int(frame), hop_sec), c, float(height)))
    events.sort(key=lambda e: e[0])
    return events


def enforce_stream_consistency(events, probs, hop_sec=HOP_SEC, window=4,
                               min_neighbors=4, majority=0.75,
                               max_gap_sec=2.0, min_prob=0.1):
    """Rewrite lone timekeeping-cymbal outliers (hat vs ride ostinato
    flip-flops) to the surface their neighbours agree on. Two-pass: flips are
    decided against the original labels so they don't cascade."""
    T = probs.shape[0]
    group = [gi for gi, e in enumerate(events) if e[1] in _GROUP_SURFACE]

    flips = {}
    for p, gi in enumerate(group):
        t, cls, _conf = events[gi]
        surf = _GROUP_SURFACE[cls]

        neigh = []
        for q in range(p - 1, max(p - 1 - window, -1), -1):
            if t - events[group[q]][0] > max_gap_sec:
                break
            neigh.append(_GROUP_SURFACE[events[group[q]][1]])
        for q in range(p + 1, min(p + 1 + window, len(group))):
            if events[group[q]][0] - t > max_gap_sec:
                break
            neigh.append(_GROUP_SURFACE[events[group[q]][1]])

        if len(neigh) < min_neighbors:
            continue

        counts = {}
        for s in neigh:
            counts[s] = counts.get(s, 0) + 1
        maj_surf = max(counts, key=counts.get)
        if maj_surf == surf or counts[maj_surf] / len(neigh) < majority:
            continue

        frame = min(max(int(round(t / hop_sec)), 0), T - 1)
        if maj_surf == "ride":
            new_cls = NAME_TO_CLASS["ride"]
        else:
            hc, ho = NAME_TO_CLASS["hihat_closed"], NAME_TO_CLASS["hihat_open"]
            new_cls = hc if probs[frame, hc] >= probs[frame, ho] else ho
        target_prob = float(probs[frame, new_cls])
        if target_prob < min_prob:
            continue
        flips[gi] = (new_cls, target_prob)

    out = [(t, *flips[gi]) if gi in flips else (t, cls, conf)
           for gi, (t, cls, conf) in enumerate(events)]
    out.sort(key=lambda e: e[0])
    return out


def write_midi(events, out_path: Path, note_length=0.05) -> None:
    """Onset events -> standard .mid file (GM drum channel). Confidence maps
    to velocity."""
    import numpy as np
    import pretty_midi

    midi = pretty_midi.PrettyMIDI()
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    for onset, cls, conf in events:
        velocity = int(np.clip(conf * 127, 1, 127))
        drums.notes.append(pretty_midi.Note(
            velocity=velocity, pitch=GM_PITCHES[cls],
            start=onset, end=onset + note_length,
        ))
    midi.instruments.append(drums)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(out_path))


# =========================================================================
# Demucs drum-stem separation (vendored from src/infer.py)
# =========================================================================

_demucs_cache = {}


def separate_drums(song_path: Path, device: str):
    """Full mix -> mono drums stem at SR, via the Demucs Python API (no stem
    file is written to disk)."""
    import librosa
    import numpy as np
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    if DEMUCS_MODEL not in _demucs_cache:
        m = get_model(DEMUCS_MODEL)
        m.eval()
        _demucs_cache[DEMUCS_MODEL] = m
    model = _demucs_cache[DEMUCS_MODEL]

    wav, _ = librosa.load(str(song_path), sr=model.samplerate, mono=False)
    if wav.ndim == 1:
        wav = np.stack([wav, wav])
    mix = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))

    ref = mix.mean(0)
    mix = (mix - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        sources = apply_model(model, mix[None], device=device,
                              shifts=1, split=True, overlap=0.25, progress=False)[0]
    drums = sources[model.sources.index("drums")] * ref.std() + ref.mean()

    mono = drums.mean(0).cpu().numpy()
    if model.samplerate != SR:
        mono = librosa.resample(mono, orig_sr=model.samplerate, target_sr=SR)
    return mono.astype(np.float32)


# =========================================================================
# FLAC metadata
# =========================================================================

REQUIRED_FIELDS = ("artist", "title", "album")

# "07 Title", "07 - Title", "07. Title", "07_Title", "(07) Title", "07-Title"
_FILENAME_RE = re.compile(r"^\s*\(?(\d{1,3})\)?\s*[-. _)]*\s*(.*\S)\s*$")
_NUM_RE = re.compile(r"^\s*(\d+)\s*(?:/\s*(\d+))?\s*$")  # "3" or "3/12"


@dataclass
class TrackMeta:
    artist: str | None = None
    title: str | None = None
    album: str | None = None
    date: str | None = None
    genre: str | None = None
    track_num: int | None = None
    track_total: int | None = None
    disc_num: int | None = None
    disc_total: int | None = None
    duration_sec: float | None = None
    from_path: list[str] = field(default_factory=list)  # fields filled from the path

    def missing_required(self) -> list[str]:
        return [f for f in REQUIRED_FIELDS if not getattr(self, f)]


def _tag(flac, *names) -> str | None:
    """First non-empty value among the given Vorbis tag names (mutagen keys
    are case-insensitive). Multi-valued tags join with '; '."""
    for name in names:
        vals = flac.get(name)
        if vals:
            joined = "; ".join(v.strip() for v in vals if v and v.strip())
            if joined:
                return joined
    return None


def _parse_num(value: str | None) -> tuple[int | None, int | None]:
    if not value:
        return None, None
    m = _NUM_RE.match(value)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2)) if m.group(2) else None


def read_flac_meta(path: Path) -> TrackMeta:
    from mutagen.flac import FLAC

    flac = FLAC(str(path))
    meta = TrackMeta()
    meta.artist = _tag(flac, "artist", "albumartist")
    meta.title = _tag(flac, "title")
    meta.album = _tag(flac, "album")
    meta.date = _tag(flac, "date", "year", "originaldate")
    meta.genre = _tag(flac, "genre")

    meta.track_num, tt = _parse_num(_tag(flac, "tracknumber"))
    meta.track_total = tt
    if meta.track_total is None:
        n, _ = _parse_num(_tag(flac, "tracktotal", "totaltracks"))
        meta.track_total = n

    meta.disc_num, dt = _parse_num(_tag(flac, "discnumber"))
    meta.disc_total = dt
    if meta.disc_total is None:
        n, _ = _parse_num(_tag(flac, "disctotal", "totaldiscs"))
        meta.disc_total = n

    meta.duration_sec = float(flac.info.length) if flac.info else None
    return meta


def fill_meta_from_path(meta: TrackMeta, path: Path, root: Path) -> TrackMeta:
    """Fill ONLY fields missing from the tags, from an
    Artist/Album/NN - Title.flac layout. Artist/album come from the two
    directories above the file when it sits at least two levels below root."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path

    track_num, title = None, path.stem
    m = _FILENAME_RE.match(path.stem)
    if m:
        track_num, title = int(m.group(1)), m.group(2)

    if len(rel.parts) >= 3:  # Artist/Album/file.flac (or deeper: nearest two)
        album_dir = path.parent.name
        artist_dir = path.parent.parent.name
        if not meta.artist and artist_dir:
            meta.artist = artist_dir
            meta.from_path.append("artist")
        if not meta.album and album_dir:
            meta.album = album_dir
            meta.from_path.append("album")
    if not meta.title and title:
        meta.title = title
        meta.from_path.append("title")
    if meta.track_num is None and track_num is not None:
        meta.track_num = track_num
        meta.from_path.append("track_num")
    return meta


# =========================================================================
# Database
# =========================================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id             INTEGER PRIMARY KEY,
    flac_path      TEXT UNIQUE NOT NULL,
    midi_path      TEXT,
    artist         TEXT NOT NULL,
    title          TEXT NOT NULL,
    album          TEXT NOT NULL,
    date           TEXT,
    genre          TEXT,
    track_num      INTEGER,
    track_total    INTEGER,
    disc_num       INTEGER,
    disc_total     INTEGER,
    duration_sec   REAL,
    meta_from_path TEXT,           -- comma list of fields inferred from the path
    n_onsets       INTEGER,
    model          TEXT,
    thresholds     TEXT,           -- JSON of per-class thresholds used
    file_size      INTEGER,
    file_mtime     REAL,
    transcribed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_tracks_artist_album ON tracks (artist, album);
CREATE TABLE IF NOT EXISTS failures (
    path   TEXT UNIQUE NOT NULL,
    reason TEXT,
    at     TEXT
);
"""


def open_db(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.executescript(SCHEMA)
    return con


def already_done(con: sqlite3.Connection, path: Path) -> bool:
    """True if this flac is in the DB with unchanged size/mtime and its
    .mid file still exists on disk."""
    row = con.execute(
        "SELECT file_size, file_mtime, midi_path FROM tracks WHERE flac_path = ?",
        (str(path),),
    ).fetchone()
    if not row or not row[2] or not Path(row[2]).exists():
        return False
    st = path.stat()
    return row[0] == st.st_size and abs(row[1] - st.st_mtime) < 1e-6


def save_track(con, flac_path: Path, midi_path: Path, meta: TrackMeta,
               n_onsets: int, model_path: str, thresholds_json: str):
    st = flac_path.stat()
    con.execute(
        """INSERT INTO tracks (flac_path, midi_path, artist, title, album,
               date, genre, track_num, track_total, disc_num, disc_total,
               duration_sec, meta_from_path, n_onsets, model, thresholds,
               file_size, file_mtime, transcribed_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(flac_path) DO UPDATE SET
               midi_path=excluded.midi_path,
               artist=excluded.artist, title=excluded.title, album=excluded.album,
               date=excluded.date, genre=excluded.genre,
               track_num=excluded.track_num, track_total=excluded.track_total,
               disc_num=excluded.disc_num, disc_total=excluded.disc_total,
               duration_sec=excluded.duration_sec,
               meta_from_path=excluded.meta_from_path,
               n_onsets=excluded.n_onsets, model=excluded.model,
               thresholds=excluded.thresholds,
               file_size=excluded.file_size, file_mtime=excluded.file_mtime,
               transcribed_at=excluded.transcribed_at""",
        (str(flac_path), str(midi_path), meta.artist, meta.title, meta.album,
         meta.date, meta.genre, meta.track_num, meta.track_total,
         meta.disc_num, meta.disc_total, meta.duration_sec,
         ",".join(meta.from_path) or None, n_onsets, model_path,
         thresholds_json, st.st_size, st.st_mtime,
         time.strftime("%Y-%m-%dT%H:%M:%S")),
    )
    con.execute("DELETE FROM failures WHERE path = ?", (str(flac_path),))
    con.commit()


def record_failure(con, path: Path, reason: str):
    con.execute(
        "INSERT INTO failures (path, reason, at) VALUES (?,?,?) "
        "ON CONFLICT(path) DO UPDATE SET reason=excluded.reason, at=excluded.at",
        (str(path), reason, time.strftime("%Y-%m-%dT%H:%M:%S")),
    )
    con.commit()


# =========================================================================
# Default model download (used when --model is not given)
# =========================================================================

def _cache_dir() -> Path:
    import os

    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "librarytranscribe"


def _download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}")

    def _progress(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            done = min(blocks * block_size, total)
            print(f"\r  {done // 2**20} / {total // 2**20} MB "
                  f"({done * 100 // total}%)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    print()
    tmp.replace(dest)


def ensure_default_model() -> str:
    """Fetch the release checkpoint (and thresholds.json) into the user
    cache directory on first use; return the local checkpoint path."""
    cache = _cache_dir() / f"v{__version__}"
    model = cache / "model.pt"
    if not model.exists():
        try:
            _download(DEFAULT_MODEL_URL, model)
        except Exception as e:
            sys.exit(f"error: could not download the model from "
                     f"{DEFAULT_MODEL_URL}: {e}\n"
                     "Pass --model /path/to/best.pt to use a local checkpoint.")
    thresholds = cache / "thresholds.json"
    if not thresholds.exists():
        try:
            _download(DEFAULT_THRESHOLDS_URL, thresholds)
        except Exception as e:
            print(f"WARNING: could not download thresholds.json ({e}); "
                  "flat 0.5 thresholds will be used")
    return str(model)


# =========================================================================
# Main
# =========================================================================

def load_thresholds(model_path: str, thresholds_path: str | None):
    """Per-class threshold array + its JSON; defaults to thresholds.json next
    to the model checkpoint, else flat 0.5."""
    import numpy as np

    path = (Path(thresholds_path).expanduser() if thresholds_path
            else Path(model_path).parent / "thresholds.json")
    if path.exists():
        thr_map = json.loads(path.read_text())
        arr = np.array([thr_map[name] for name in CLASS_NAMES])
        print(f"Using calibrated thresholds from {path}")
        return arr, json.dumps(thr_map)
    if thresholds_path:
        sys.exit(f"error: thresholds file not found: {thresholds_path}")
    print("WARNING: no thresholds.json next to model checkpoint; using flat 0.5")
    return 0.5, json.dumps({name: 0.5 for name in CLASS_NAMES})


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Transcribe a FLAC library to drum MIDI (into <root>-midi) "
                    "and build a SQLite database of metadata + paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("root", help="Directory to scan recursively for .flac files")
    parser.add_argument("--db", default="drumlibrary.db", help="SQLite database path")
    parser.add_argument("--model", default=None,
                        help="Model checkpoint (.pt); default: download the "
                             "release model into the user cache on first use")
    parser.add_argument("--thresholds", default=None,
                        help="thresholds.json (default: next to the model checkpoint)")
    parser.add_argument("--force", action="store_true",
                        help="Re-transcribe files already in the database")
    parser.add_argument("--no-stream-vote", action="store_true",
                        help="Disable stream-consistency voting (on by default)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N files (for smoke tests)")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"error: not a directory: {root}")
    midi_root = root / "midi"

    files = sorted(p for p in root.rglob("*") if p.suffix.lower() == ".flac")
    print(f"Found {len(files)} .flac files under {root}")
    print(f"MIDI output tree: {midi_root}")
    if args.limit:
        files = files[: args.limit]

    con = open_db(Path(args.db).expanduser())

    # ---- metadata pass -------------------------------------------------
    ready: list[tuple[Path, TrackMeta]] = []
    n_skipped_meta = 0
    for path in files:
        try:
            meta = read_flac_meta(path)
        except Exception as e:
            print(f"  UNREADABLE {path}: {e}")
            record_failure(con, path, f"metadata: {e}")
            n_skipped_meta += 1
            continue
        meta = fill_meta_from_path(meta, path, root)
        missing = meta.missing_required()
        if missing:
            print(f"  SKIP (missing {', '.join(missing)}): {path.relative_to(root)}")
            record_failure(con, path, f"missing required metadata: {', '.join(missing)}")
            n_skipped_meta += 1
            continue
        ready.append((path, meta))

    print(f"{len(ready)} tracks with sufficient metadata, "
          f"{n_skipped_meta} skipped/unreadable")

    # ---- model setup ---------------------------------------------------
    import torch

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_path = (str(Path(args.model).expanduser().resolve()) if args.model
                  else ensure_default_model())
    print(f"Loading model {model_path} on {device}...")
    model = load_model_from_checkpoint(model_path, device)
    thresholds, thresholds_json = load_thresholds(model_path, args.thresholds)

    # ---- transcription pass -------------------------------------------
    n_done = n_skip = n_fail = 0
    for i, (path, meta) in enumerate(ready, 1):
        rel = path.relative_to(root)
        if not args.force and already_done(con, path):
            n_skip += 1
            continue
        print(f"[{i}/{len(ready)}] {rel}")
        midi_path = (midi_root / rel).with_suffix(".mid")
        try:
            audio = separate_drums(path, str(device))
            probs = predict_track(model, audio, device)
            events = decode_onsets(probs, thresholds=thresholds)
            if not args.no_stream_vote:
                events = enforce_stream_consistency(events, probs)
            write_midi(events, midi_path)
        except KeyboardInterrupt:
            print("\nInterrupted -- progress so far is saved; rerun to resume.")
            break
        except Exception as e:
            print(f"  FAILED: {e}")
            record_failure(con, path, f"transcription: {e}")
            n_fail += 1
            continue

        save_track(con, path, midi_path, meta, len(events),
                   model_path, thresholds_json)
        print(f"  {len(events)} onsets -> {midi_path}")
        n_done += 1

    print(f"\nDone: {n_done} transcribed, {n_skip} already up to date, "
          f"{n_fail} failed, {n_skipped_meta} skipped on metadata. DB: {args.db}")


if __name__ == "__main__":
    main()
