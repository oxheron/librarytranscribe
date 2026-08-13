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
# The audfprint-compatible implementation below intentionally reuses numpy,
# scipy and librosa instead of depending on audfprint2: that package requires
# numpy >= 2.3, which is incompatible with the Intel-mac torch wheel.
# On Intel macs use Python <= 3.12 (the last torch 2.2.2 wheel is cp312).
"""
Build drum-transcription and audio-fingerprint databases from a FLAC library.

Recursively finds .flac files under a root directory, reads their metadata,
transcribes each one to drum MIDI with a trained ADT model (+ calibrated
per-class thresholds), writes the .mid files into a "midi" subdirectory of
the root that mirrors the library layout, records metadata + both file paths
in SQLite, and builds an audfprint-compatible landmark database from the
original mixes.

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
  librarytranscribe /music/files -o drums.db [--model runs/full/best.pt]

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
  sqlite3 /music/files/drums.db "SELECT artist, album, title, n_onsets FROM tracks"
  sqlite3 /music/files/drums.db "SELECT flac_path, midi_path FROM tracks WHERE artist='X'"
  audfprint match --dbase /music/files/audfprint.pklz --shifts 4 query.wav
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

__version__ = "0.2.0"

# Where the default checkpoint is fetched from when --model is not given.
# Publishing a new model means uploading model.pt + thresholds.json to a
# GitHub release tagged v<__version__> (see README.md).
_RELEASE_URL = ("https://github.com/oxheron/librarytranscribe/releases/download/"
                f"v{__version__}")
DEFAULT_MODEL_URL = f"{_RELEASE_URL}/model.pt"
DEFAULT_THRESHOLDS_URL = f"{_RELEASE_URL}/thresholds.json"


# =========================================================================
# audfprint settings
# =========================================================================

# These match the requested database/query profile.  A database shift count
# of zero means one unshifted analysis pass; query shifts are documented and
# embedded in the database parameters for clients to discover.
AUDFPRINT_SAMPLERATE = 11025
AUDFPRINT_DENSITY = 70.0
AUDFPRINT_FANOUT = 8
AUDFPRINT_HASHBITS = 20
AUDFPRINT_BUCKETSIZE = 256
AUDFPRINT_MAXTIMEBITS = 17
AUDFPRINT_DB_SHIFTS = 0
AUDFPRINT_QUERY_SHIFTS = 4
AUDFPRINT_DB_DEFAULT = "audfprint.pklz"


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
# audfprint-compatible landmark fingerprints
#
# The implementation follows Dan Ellis's audfprint landmark/hash layout and
# pickle database fields.  It lives here instead of pulling in audfprint2,
# whose numpy>=2.3 requirement conflicts with torch 2.2 on Intel macOS.  The
# database can be opened by audfprint/audfprint2; its track names are the
# absolute FLAC paths used by this tool.
#
# Copyright (c) 2014-2015 Dan Ellis, Columbia University, and Google.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions: The above
# copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
# WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# =========================================================================

_AUDFPRINT_HT_VERSION = 20170724
_AUDFPRINT_F1_BITS = 8
_AUDFPRINT_DF_BITS = 6
_AUDFPRINT_DT_BITS = 6
_AUDFPRINT_B1_MASK = (1 << _AUDFPRINT_F1_BITS) - 1
_AUDFPRINT_B1_SHIFT = _AUDFPRINT_DF_BITS + _AUDFPRINT_DT_BITS
_AUDFPRINT_DF_MASK = (1 << _AUDFPRINT_DF_BITS) - 1
_AUDFPRINT_DF_SHIFT = _AUDFPRINT_DT_BITS
_AUDFPRINT_DT_MASK = (1 << _AUDFPRINT_DT_BITS) - 1


def _bits_for_power_of_two(value: int) -> int:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"value must be a positive power of two, not {value}")
    return value.bit_length() - 1


class _AudfprintHashTable:
    """The audfprint pickle fields needed to build and update a database."""

    def __init__(self, hashbits=AUDFPRINT_HASHBITS,
                 depth=AUDFPRINT_BUCKETSIZE,
                 maxtime=(1 << AUDFPRINT_MAXTIMEBITS)):
        import numpy as np

        self.hashbits = hashbits
        self.depth = depth
        self.maxtimebits = _bits_for_power_of_two(maxtime)
        self.table = np.zeros((1 << hashbits, depth), dtype=np.uint32)
        self.counts = np.zeros(1 << hashbits, dtype=np.int32)
        self.names: list[str | None] = []
        self.hashesperid = np.zeros(0, dtype=np.uint32)
        self.params: dict[str, object] = {}
        self.ht_version = _AUDFPRINT_HT_VERSION
        self.dirty = True

    def __setstate__(self, state) -> None:
        """Accept regular and slots-style state from upstream pickles."""
        if isinstance(state, tuple) and len(state) == 2:
            dict_state, slot_state = state
            if dict_state:
                self.__dict__.update(dict_state)
            if slot_state:
                self.__dict__.update(slot_state)
        else:
            self.__dict__.update(state)

    def _name_to_id(self, name: str, add_if_missing=False) -> int:
        import numpy as np

        if name not in self.names:
            if not add_if_missing:
                raise ValueError(f"name {name} not found")
            try:
                track_id = self.names.index(None)
                self.names[track_id] = name
                self.hashesperid[track_id] = 0
            except ValueError:
                track_id = len(self.names)
                max_tracks = (1 << (32 - self.maxtimebits)) - 1
                if track_id >= max_tracks:
                    raise ValueError(
                        f"audfprint database is full ({max_tracks} tracks with "
                        f"maxtimebits={self.maxtimebits})")
                self.names.append(name)
                self.hashesperid = np.append(
                    self.hashesperid, np.array([0], dtype=np.uint32))
        return self.names.index(name)

    def store(self, name: str, time_hash_pairs) -> None:
        """Store an iterable of audfprint ``(frame, hash)`` pairs."""
        import random

        track_id = self._name_to_id(name, add_if_missing=True)
        hashmask = (1 << self.hashbits) - 1
        timemask = (1 << self.maxtimebits) - 1
        id_value = (track_id + 1) << self.maxtimebits
        n_hashes = 0
        for time_frame, hash_value in time_hash_pairs:
            hash_value = int(hash_value) & hashmask
            count = int(self.counts[hash_value])
            value = id_value + (int(time_frame) & timemask)
            if count < self.depth:
                self.table[hash_value, count] = value
            else:
                slot = random.randint(0, count)
                if slot < self.depth:
                    self.table[hash_value, slot] = value
            self.counts[hash_value] = count + 1
            n_hashes += 1
        self.hashesperid[track_id] += n_hashes
        self.dirty = True

    def remove(self, name: str) -> None:
        """Remove a track before replacing fingerprints for a changed file."""
        import numpy as np

        track_id = self._name_to_id(name)
        encoded_id = track_id + 1
        # Work in chunks: vectorizing the whole 400 MiB table would allocate
        # another ~100 MiB boolean array, while visiting every bucket in pure
        # Python makes replacement painfully slow for established databases.
        chunk_rows = 4096
        for start in range(0, len(self.counts), chunk_rows):
            stop = min(start + chunk_rows, len(self.counts))
            id_mask = ((self.table[start:stop] >> self.maxtimebits)
                       == encoded_id)
            for local_row in np.nonzero(np.any(id_mask, axis=1))[0]:
                hash_value = start + int(local_row)
                stored = min(self.depth, int(self.counts[hash_value]))
                values = self.table[hash_value, :stored]
                keep = (values >> self.maxtimebits) != encoded_id
                remaining = values[keep]
                self.table[hash_value, :] = 0
                self.table[hash_value, :len(remaining)] = remaining
                # This mirrors audfprint: after removal, previously dropped
                # entries cannot be recovered, so the exact stored count wins.
                self.counts[hash_value] = len(remaining)
        self.names[track_id] = None
        self.hashesperid[track_id] = 0
        self.dirty = True


class _AudfprintUnpickler:
    """Factory for an unpickler that accepts upstream HashTable class paths."""

    @staticmethod
    def load(file_obj):
        import pickle

        class CompatibleUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == "HashTable" and module in {
                    "hash_table", "audfprint2.core.hash_table",
                }:
                    return _AudfprintHashTable
                if name == "_AudfprintHashTable" and module in {
                    "__main__", "librarytranscribe",
                }:
                    return _AudfprintHashTable
                return super().find_class(module, name)

        return CompatibleUnpickler(file_obj, encoding="latin1").load()


def _audfprint_params() -> dict[str, int | float]:
    return {
        "samplerate": AUDFPRINT_SAMPLERATE,
        "density": AUDFPRINT_DENSITY,
        "fanout": AUDFPRINT_FANOUT,
        "hashbits": AUDFPRINT_HASHBITS,
        "bucketsize": AUDFPRINT_BUCKETSIZE,
        "maxtimebits": AUDFPRINT_MAXTIMEBITS,
        "db_shifts": AUDFPRINT_DB_SHIFTS,
        "query_shifts": AUDFPRINT_QUERY_SHIFTS,
    }


def _open_audfprint_db(path: Path) -> _AudfprintHashTable:
    import gzip

    if not path.exists():
        table = _AudfprintHashTable()
        table.params.update(_audfprint_params())
        return table
    opener = open if path.suffix.lower() == ".pkl" else gzip.open
    with opener(path, "rb") as file_obj:
        loaded = _AudfprintUnpickler.load(file_obj)
    required = ("hashbits", "depth", "maxtimebits", "table", "counts",
                "names", "hashesperid", "params")
    if any(not hasattr(loaded, field) for field in required):
        raise ValueError(f"not an audfprint pickle database: {path}")
    if isinstance(loaded, _AudfprintHashTable):
        table = loaded
    else:
        # New databases use a standard-library SimpleNamespace payload so
        # upstream audfprint can unpickle them without librarytranscribe being
        # installed.  Turn that attribute bag back into our mutable table.
        table = _AudfprintHashTable.__new__(_AudfprintHashTable)
        table.__dict__.update(vars(loaded))
    expected = (AUDFPRINT_HASHBITS, AUDFPRINT_BUCKETSIZE,
                AUDFPRINT_MAXTIMEBITS)
    actual = (int(table.hashbits), int(table.depth), int(table.maxtimebits))
    if actual != expected:
        raise ValueError(
            f"audfprint database geometry is {actual}, expected {expected}: {path}")
    stored_sr = table.params.get("samplerate")
    if stored_sr is not None and int(stored_sr) != AUDFPRINT_SAMPLERATE:
        raise ValueError(
            f"audfprint database samplerate is {stored_sr}, expected "
            f"{AUDFPRINT_SAMPLERATE}: {path}")
    table.params.update(_audfprint_params())
    table.dirty = False
    return table


def _save_audfprint_db(table: _AudfprintHashTable, path: Path) -> None:
    import gzip
    import pickle
    from types import SimpleNamespace

    if path.suffix.lower() not in {".pkl", ".pklz"}:
        raise ValueError("audfprint database must end in .pkl or .pklz")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".part")
    opener = open if path.suffix.lower() == ".pkl" else gzip.open
    try:
        with opener(temp_path, "wb") as file_obj:
            # Both upstream implementations load a temporary object's fields
            # into their own HashTable instance.  A SimpleNamespace supplies
            # those fields without embedding a librarytranscribe class path.
            payload = SimpleNamespace(**table.__dict__)
            pickle.dump(payload, file_obj, pickle.HIGHEST_PROTOCOL)
        temp_path.replace(path)
        table.dirty = False
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _audfprint_locmax(vector, indices=False):
    import numpy as np

    neighbors = np.zeros(len(vector) + 1, dtype=bool)
    neighbors[0] = True
    neighbors[1:-1] = np.greater_equal(vector[1:], vector[:-1])
    maxima = neighbors[:-1] & ~neighbors[1:]
    return np.nonzero(maxima)[0] if indices else maxima


def _audfprint_stft(signal, n_fft, hop_length, window):
    """audfprint's reflect-padded, non-copying STFT implementation."""
    import numpy as np

    padded = np.pad(signal, n_fft // 2, mode="reflect")
    frame_count = 1 + (len(padded) - len(window)) // hop_length
    shape = (frame_count, len(window))
    strides = (padded.strides[0] * hop_length, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        padded, shape=shape, strides=strides)
    return np.fft.rfft(frames * window, n_fft).transpose()


class _AudfprintAnalyzer:
    """Convert mono 11.025 kHz audio arrays to audfprint hashes."""

    def __init__(self):
        import numpy as np

        self.density = AUDFPRINT_DENSITY
        self.n_fft = 512
        self.n_hop = 256
        self.f_sd = 30.0
        self.maxpksperframe = 5
        self.maxpairsperpeak = AUDFPRINT_FANOUT
        self.targetdf = 31
        self.mindt = 2
        self.targetdt = 63
        self._spread_width = None
        self._spread_len = None
        self._spread_values = np.array([])

    def _spread_peaks(self, peaks, npoints=None, width=4.0, base=None):
        import numpy as np

        if base is None:
            if npoints is None:
                raise ValueError("npoints is required without a base vector")
            vector = np.zeros(npoints)
        else:
            npoints = len(base)
            vector = np.copy(base)
        if width != self._spread_width or npoints != self._spread_len:
            self._spread_width = width
            self._spread_len = npoints
            self._spread_values = np.exp(
                -0.5 * (np.arange(-npoints, npoints + 1) / width) ** 2)
        for position, value in peaks:
            indexes = np.arange(npoints) + npoints - position
            vector = np.maximum(vector, value * self._spread_values[indexes])
        return vector

    def _spread_vector_peaks(self, vector, width=4.0):
        peaks = _audfprint_locmax(vector, indices=True)
        return self._spread_peaks(
            zip(peaks, vector[peaks]), npoints=len(vector), width=width)

    def _forward_prune(self, spectrogram, decay):
        import numpy as np

        _rows, columns = np.shape(spectrogram)
        threshold = self._spread_vector_peaks(
            np.max(spectrogram[:, :min(10, columns)], axis=1), self.f_sd)
        peaks = np.zeros(np.shape(spectrogram), dtype=bool)
        spread_points = len(threshold)
        spread_values = self._spread_values
        for column in range(columns):
            values = spectrogram[:, column]
            candidates = np.nonzero(
                _audfprint_locmax(values) & (values > threshold))[0]
            ranked = sorted(zip(values[candidates], candidates), reverse=True)
            for value, position in ranked[:self.maxpksperframe]:
                threshold = np.maximum(
                    threshold,
                    value * spread_values[
                        spread_points - position:2 * spread_points - position],
                )
                peaks[position, column] = True
            threshold *= decay
        return peaks

    def _backward_prune(self, spectrogram, peaks, decay):
        import numpy as np

        columns = np.shape(spectrogram)[1]
        threshold = self._spread_vector_peaks(spectrogram[:, -1], self.f_sd)
        for column in range(columns, 0, -1):
            candidates = np.nonzero(peaks[:, column - 1])[0]
            values = spectrogram[candidates, column - 1]
            for value, position in sorted(
                    zip(values, candidates), reverse=True):
                if value >= threshold[position]:
                    threshold = self._spread_peaks(
                        [(position, value)], base=threshold, width=self.f_sd)
                    if column < columns:
                        peaks[position, column] = False
                else:
                    peaks[position, column - 1] = False
            threshold *= decay
        return peaks

    def _find_peaks(self, audio):
        import numpy as np
        from scipy.signal import lfilter

        if len(audio) == 0:
            return []
        decay = 1 - 0.01 * (
            self.density * np.sqrt(self.n_hop / 352.8) / 35)
        window = np.hanning(self.n_fft + 2)[1:-1]
        spectrogram = np.abs(_audfprint_stft(
            audio, self.n_fft, self.n_hop, window))
        maximum = np.max(spectrogram)
        if maximum > 0:
            spectrogram = np.log(np.maximum(spectrogram, maximum / 1e6))
            spectrogram -= np.mean(spectrogram)
        spectrogram = np.array([
            lfilter([1, -1], [1, -0.98], row) for row in spectrogram
        ])[:-1, :]
        peaks = self._forward_prune(spectrogram, decay)
        peaks = self._backward_prune(spectrogram, peaks, decay)
        result = []
        for column in range(np.shape(spectrogram)[1]):
            result.extend(
                (column, int(bin_)) for bin_ in np.nonzero(peaks[:, column])[0])
        return result

    def _peaks_to_landmarks(self, peaks):
        if not peaks:
            return []
        columns = peaks[-1][0] + 1
        peaks_at = [[] for _ in range(columns)]
        for column, bin_ in peaks:
            peaks_at[column].append(bin_)
        landmarks = []
        for column in range(columns):
            for peak in peaks_at[column]:
                pairs = 0
                for later in range(
                        column + self.mindt,
                        min(columns, column + self.targetdt)):
                    if pairs >= self.maxpairsperpeak:
                        break
                    for later_peak in peaks_at[later]:
                        if (abs(later_peak - peak) < self.targetdf and
                                pairs < self.maxpairsperpeak):
                            landmarks.append(
                                (column, peak, later_peak, later - column))
                            pairs += 1
        return landmarks

    def hashes(self, audio):
        import numpy as np

        landmarks = np.asarray(self._peaks_to_landmarks(
            self._find_peaks(audio)), dtype=np.int32)
        if not len(landmarks):
            return np.zeros((0, 2), dtype=np.int32)
        hashes = np.zeros((len(landmarks), 2), dtype=np.int32)
        hashes[:, 0] = landmarks[:, 0]
        hashes[:, 1] = (
            ((landmarks[:, 1] & _AUDFPRINT_B1_MASK) << _AUDFPRINT_B1_SHIFT)
            | (((landmarks[:, 2] - landmarks[:, 1]) & _AUDFPRINT_DF_MASK)
               << _AUDFPRINT_DF_SHIFT)
            | (landmarks[:, 3] & _AUDFPRINT_DT_MASK)
        )
        packed = ((hashes[:, 0].astype(np.uint64) << 32)
                  | hashes[:, 1].astype(np.uint32))
        unique = np.sort(np.unique(packed))
        return np.column_stack((unique >> 32, unique & 0xFFFFFFFF)).astype(
            np.int32)


def _fingerprint_file(path: Path, analyzer: _AudfprintAnalyzer):
    import librosa
    import numpy as np

    audio, _ = librosa.load(
        str(path), sr=AUDFPRINT_SAMPLERATE, mono=True, dtype=np.float32)
    return analyzer.hashes(audio)


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
CREATE TABLE IF NOT EXISTS fingerprints (
    flac_path       TEXT PRIMARY KEY,
    file_size       INTEGER NOT NULL,
    file_mtime      REAL NOT NULL,
    n_hashes        INTEGER NOT NULL,
    fingerprinted_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fingerprint_failures (
    path   TEXT PRIMARY KEY,
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


def fingerprint_already_done(con: sqlite3.Connection, path: Path,
                             known_names: set[str]) -> bool:
    row = con.execute(
        "SELECT file_size, file_mtime FROM fingerprints WHERE flac_path = ?",
        (str(path),),
    ).fetchone()
    if not row or str(path) not in known_names:
        return False
    stat = path.stat()
    return row[0] == stat.st_size and abs(row[1] - stat.st_mtime) < 1e-6


def _save_fingerprint_records(con: sqlite3.Connection, records) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    con.executemany(
        """INSERT INTO fingerprints
               (flac_path, file_size, file_mtime, n_hashes, fingerprinted_at)
           VALUES (?,?,?,?,?)
           ON CONFLICT(flac_path) DO UPDATE SET
               file_size=excluded.file_size,
               file_mtime=excluded.file_mtime,
               n_hashes=excluded.n_hashes,
               fingerprinted_at=excluded.fingerprinted_at""",
        [(str(path), stat.st_size, stat.st_mtime, n_hashes, now)
         for path, stat, n_hashes in records],
    )
    con.executemany(
        "DELETE FROM fingerprint_failures WHERE path = ?",
        [(str(path),) for path, _stat, _n_hashes in records],
    )
    con.commit()


def _record_fingerprint_failure(con: sqlite3.Connection, path: Path,
                                reason: str) -> None:
    con.execute(
        "INSERT INTO fingerprint_failures (path, reason, at) VALUES (?,?,?) "
        "ON CONFLICT(path) DO UPDATE SET reason=excluded.reason, at=excluded.at",
        (str(path), reason, time.strftime("%Y-%m-%dT%H:%M:%S")),
    )
    con.commit()


def build_fingerprint_database(con: sqlite3.Connection, files: list[Path],
                               db_path: Path, force=False):
    """Incrementally fingerprint files and atomically update the pickle DB."""
    try:
        table = _open_audfprint_db(db_path)
    except Exception as exc:
        sys.exit(f"error: could not open audfprint database {db_path}: {exc}")

    known_names = {name for name in table.names if name is not None}
    analyzer = _AudfprintAnalyzer()
    records = []
    n_done = n_skip = n_fail = 0
    interrupted = False

    for index, path in enumerate(files, 1):
        name = str(path)
        if not force and fingerprint_already_done(con, path, known_names):
            n_skip += 1
            continue
        print(f"[fingerprint {index}/{len(files)}] {path.name}")
        try:
            hashes = _fingerprint_file(path, analyzer)
            # Analyze first so a read/analysis failure leaves any older entry
            # intact.  Replacing only after success avoids duplicate hashes.
            if name in known_names:
                table.remove(name)
                known_names.remove(name)
            pairs = [(int(row[0]), int(row[1])) for row in hashes]
            table.store(name, pairs)
            known_names.add(name)
            records.append((path, path.stat(), len(pairs)))
            n_done += 1
            print(f"  {len(pairs)} hashes")
        except KeyboardInterrupt:
            print("\nInterrupted -- saving fingerprint progress so far.")
            interrupted = True
            break
        except Exception as exc:
            print(f"  FINGERPRINT FAILED: {exc}")
            _record_fingerprint_failure(con, path, str(exc))
            n_fail += 1

    if table.dirty:
        print(f"Saving audfprint database: {db_path}")
        try:
            _save_audfprint_db(table, db_path)
        except Exception as exc:
            sys.exit(f"error: could not save audfprint database {db_path}: {exc}")
        # Commit the incremental index only after the fingerprint database is
        # safely in place.  A crash can therefore cause re-analysis, not loss.
        _save_fingerprint_records(con, records)

    return n_done, n_skip, n_fail, interrupted


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
        description="Transcribe a FLAC library to drum MIDI (into <root>/midi) "
                    "and build SQLite + audfprint databases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("root", help="Directory to scan recursively for .flac files")
    parser.add_argument("-o", "--output", "--db", dest="db",
                        default="drumlibrary.db",
                        help="SQLite database name/path (relative paths go under root)")
    parser.add_argument("--model", default=None,
                        help="Model checkpoint (.pt); default: download the "
                             "release model into the user cache on first use")
    parser.add_argument("--thresholds", default=None,
                        help="thresholds.json (default: next to the model checkpoint)")
    parser.add_argument("--audfprint-db", "--fingerprint-db",
                        dest="fingerprint_db", default=AUDFPRINT_DB_DEFAULT,
                        help="audfprint .pklz database (relative paths go under root)")
    parser.add_argument("--midi-dir", default=None,
                        help="MIDI output tree root (default: <root>/midi)")
    parser.add_argument("--no-audfprint", action="store_true",
                        help="Skip audfprint database creation/update")
    parser.add_argument("--force", action="store_true",
                        help="Re-transcribe and re-fingerprint unchanged files")
    parser.add_argument("--no-stream-vote", action="store_true",
                        help="Disable stream-consistency voting (on by default)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N files (for smoke tests)")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"error: not a directory: {root}")
    midi_root = (Path(args.midi_dir).expanduser().resolve() if args.midi_dir
                 else root / "midi")
    db_path = Path(args.db).expanduser()
    if not db_path.is_absolute():
        db_path = root / db_path
    db_path = db_path.resolve()
    fingerprint_db_path = Path(args.fingerprint_db).expanduser()
    if not fingerprint_db_path.is_absolute():
        fingerprint_db_path = root / fingerprint_db_path
    fingerprint_db_path = fingerprint_db_path.resolve()
    if (not args.no_audfprint and
            fingerprint_db_path.suffix.lower() not in {".pkl", ".pklz"}):
        sys.exit("error: audfprint database must end in .pkl or .pklz")
    if not args.no_audfprint and fingerprint_db_path == db_path:
        sys.exit("error: SQLite and audfprint databases must use different paths")
    # The MIDI writer creates its tree as it goes, but sqlite3.connect fails
    # if the database's directory does not exist yet (e.g. redirected output).
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_audfprint:
        fingerprint_db_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in root.rglob("*") if p.suffix.lower() == ".flac")
    print(f"Found {len(files)} .flac files under {root}")
    print(f"MIDI output tree: {midi_root}")
    print(f"Database: {db_path}")
    if not args.no_audfprint:
        print(f"audfprint database: {fingerprint_db_path}")
        print(f"audfprint settings: samplerate={AUDFPRINT_SAMPLERATE} "
              f"density={AUDFPRINT_DENSITY:g} fanout={AUDFPRINT_FANOUT} "
              f"hashbits={AUDFPRINT_HASHBITS} bucketsize={AUDFPRINT_BUCKETSIZE} "
              f"maxtimebits={AUDFPRINT_MAXTIMEBITS} "
              f"DB shifts={AUDFPRINT_DB_SHIFTS} "
              f"query shifts={AUDFPRINT_QUERY_SHIFTS}")
    if args.limit:
        files = files[: args.limit]

    con = open_db(db_path)

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

    # Fingerprints use the original mixes and do not require music metadata,
    # so every discovered/readable-by-librosa FLAC is included even if it was
    # not eligible for drum transcription.
    fp_done = fp_skip = fp_fail = 0
    if not args.no_audfprint:
        fp_done, fp_skip, fp_fail, interrupted = build_fingerprint_database(
            con, files, fingerprint_db_path, force=args.force)
        print(f"Fingerprints: {fp_done} added/updated, {fp_skip} already up to "
              f"date, {fp_fail} failed. DB: {fingerprint_db_path}")
        if interrupted:
            con.close()
            return

    if not ready:
        print(f"\nDone: no tracks eligible for MIDI transcription. DB: {db_path}")
        con.close()
        return

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
          f"{n_fail} failed, {n_skipped_meta} skipped on metadata. DB: {db_path}")
    if not args.no_audfprint:
        print(f"audfprint: {fp_done} added/updated, {fp_skip} already up to date, "
              f"{fp_fail} failed. DB: {fingerprint_db_path}")
    con.close()


if __name__ == "__main__":
    main()
