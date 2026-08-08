import gzip
import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

import librarytranscribe as lt

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None


class CoreTests(unittest.TestCase):
    def test_audfprint_profile(self):
        self.assertEqual(
            lt._audfprint_params(),
            {
                "samplerate": 11025,
                "density": 70.0,
                "fanout": 8,
                "hashbits": 20,
                "bucketsize": 100,
                "maxtimebits": 17,
                "db_shifts": 0,
                "query_shifts": 4,
            },
        )

    def test_power_of_two_bits(self):
        self.assertEqual(lt._bits_for_power_of_two(1 << 17), 17)
        with self.assertRaises(ValueError):
            lt._bits_for_power_of_two(17)

    def test_inline_and_wheel_dependencies_share_mac_intel_pins(self):
        if tomllib is None:
            self.skipTest("tomllib is built into Python 3.11+")
        project_root = Path(__file__).parent.parent
        project = tomllib.loads((project_root / "pyproject.toml").read_text())
        source = (project_root / "librarytranscribe.py").read_text()
        mac_dependencies = [
            dependency for dependency in project["project"]["dependencies"]
            if "sys_platform == 'darwin'" in dependency
        ]
        self.assertGreaterEqual(len(mac_dependencies), 5)
        for dependency in mac_dependencies:
            self.assertIn(f'"{dependency}"', source)

    def test_schema_tracks_fingerprint_state_separately(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            connection = lt.open_db(Path(temp_dir) / "library.db")
            tables = {
                row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")
            }
            self.assertTrue({
                "tracks", "failures", "fingerprints",
                "fingerprint_failures",
            }.issubset(tables))
            connection.close()

    def test_fingerprint_incremental_check_uses_file_stat_and_db_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "track.flac"
            path.write_bytes(b"fake")
            connection = sqlite3.connect(":memory:")
            connection.executescript(lt.SCHEMA)
            stat = path.stat()
            connection.execute(
                "INSERT INTO fingerprints VALUES (?,?,?,?,?)",
                (str(path), stat.st_size, stat.st_mtime, 12, "now"),
            )
            self.assertTrue(lt.fingerprint_already_done(
                connection, path, {str(path)}))
            self.assertFalse(lt.fingerprint_already_done(
                connection, path, set()))
            path.write_bytes(b"changed")
            self.assertFalse(lt.fingerprint_already_done(
                connection, path, {str(path)}))
            connection.close()

    def test_hash_table_replaces_a_track_without_removing_others(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")

        table = lt._AudfprintHashTable(hashbits=4, depth=5, maxtime=1 << 8)
        table.store("first.flac", [(1, 2), (2, 3)])
        table.store("second.flac", [(7, 2)])
        table.remove("first.flac")
        self.assertEqual(table.names, [None, "second.flac"])
        self.assertEqual(int(table.counts[2]), 1)
        self.assertEqual(
            int(table.table[2, 0] >> table.maxtimebits), 2)

        table.store("first.flac", [(9, 4)])
        self.assertEqual(table.names, ["first.flac", "second.flac"])
        self.assertTrue(np.array_equal(table.hashesperid, [1, 1]))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "small.pklz"
            lt._save_audfprint_db(table, path)
            with gzip.open(path, "rb") as file_obj:
                loaded = lt._AudfprintUnpickler.load(file_obj)
            self.assertEqual(loaded.__class__.__module__, "types")
            self.assertEqual(loaded.__class__.__name__, "SimpleNamespace")
            self.assertEqual(loaded.names, table.names)
            self.assertTrue(np.array_equal(loaded.table, table.table))

    def test_hashes_match_audfprint_reference_fixture(self):
        """Fixture verified byte-for-byte against audfprint2 0.1.2."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy/scipy are not installed")

        sample_rate = lt.AUDFPRINT_SAMPLERATE
        rng = np.random.default_rng(20260807)
        time_axis = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
        audio = (
            0.2 * np.sin(2 * np.pi * 220 * time_axis)
            + 0.15 * np.sin(
                2 * np.pi * (330 + 40 * time_axis) * time_axis)
            + 0.01 * rng.standard_normal(len(time_axis))
        ).astype(np.float32)
        for start in range(0, len(audio), sample_rate // 2):
            stop = min(start + 100, len(audio))
            audio[start:stop] += np.hanning((stop - start) * 2)[:stop - start]

        analyzer = lt._AudfprintAnalyzer()
        peaks = analyzer._find_peaks(audio)
        hashes = analyzer.hashes(audio)
        self.assertEqual(len(peaks), 19)
        self.assertEqual(len(hashes), 29)
        self.assertEqual(
            hashlib.sha256(hashes.tobytes()).hexdigest(),
            "4e8ee38fe338b2d7d5c114f5676c81abca2bec8967e0e634fe0a5158e1461789",
        )


if __name__ == "__main__":
    unittest.main()
