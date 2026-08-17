# Release notes

## v0.3.0

- Add `--midi-dir` so MIDI output can be redirected independently of the
  source library, SQLite database, and audfprint database.
- Create parent directories automatically for redirected SQLite, MIDI, and
  audfprint outputs.
- Increase the audfprint bucket size to 256 for library-scale fingerprint
  densities.
- Checkpoint the audfprint database every 500 new fingerprints, limiting the
  amount of work lost if a long run is interrupted.
