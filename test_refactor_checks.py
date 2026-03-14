import importlib.util
import sys
import types


class _FakeNP(types.SimpleNamespace):
    @staticmethod
    def array(values, dtype=float):
        return list(values)

    @staticmethod
    def zeros(n, dtype=float):
        return [0.0] * n

    @staticmethod
    def eye(n, dtype=float):
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    @staticmethod
    def outer(a, b):
        return [[float(x) * float(y) for y in b] for x in a]

    @staticmethod
    def trace(m):
        return float(sum(m[i][i] for i in range(min(len(m), len(m[0])))))

    @staticmethod
    def tanh(x):
        import math

        return math.tanh(x)

    @staticmethod
    def median(vals):
        vals = sorted(float(v) for v in vals)
        n = len(vals)
        if n == 0:
            return 0.0
        if n % 2:
            return vals[n // 2]
        return 0.5 * (vals[n // 2 - 1] + vals[n // 2])

    @staticmethod
    def std(vals):
        vals = [float(v) for v in vals]
        if not vals:
            return 0.0
        mean = sum(vals) / len(vals)
        return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5

    @staticmethod
    def clip(v, lo, hi):
        return max(lo, min(hi, v))


sys.modules.setdefault("numpy", _FakeNP())
sys.modules.setdefault("irsdk", types.SimpleNamespace(IRSDK=lambda: None))
qt_ns = types.SimpleNamespace(
    QtCore=types.SimpleNamespace(Qt=types.SimpleNamespace(LeftButton=1, AlignLeft=0, AlignTop=0, Horizontal=0, FramelessWindowHint=0, Tool=0, WindowStaysOnTopHint=0), QPoint=object, QTimer=object),
    QtGui=types.SimpleNamespace(QMouseEvent=object),
    QtWidgets=types.SimpleNamespace(QWidget=object, QDialog=object),
)
sys.modules.setdefault("PyQt5", qt_ns)
sys.modules.setdefault("PyQt5.QtCore", qt_ns.QtCore)
sys.modules.setdefault("PyQt5.QtGui", qt_ns.QtGui)
sys.modules.setdefault("PyQt5.QtWidgets", qt_ns.QtWidgets)

import tempfile
import time
from pathlib import Path


MODULE_PATH = Path(__file__).parent / "iracing_tire_overlay (16).py"
spec = importlib.util.spec_from_file_location("overlay_mod", MODULE_PATH)
overlay = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = overlay
spec.loader.exec_module(overlay)


def make_snap(**kw):
    base = dict(
        session_time=0.0,
        lap=0,
        lap_dist_pct=0.0,
        on_pit_road=False,
        speed_mps=50.0,
        lat_accel=1.0,
        long_accel=0.0,
        steering=0.0,
        track_temp=30.0,
        air_temp=20.0,
        humidity=40.0,
        wear={t: 100.0 for t in overlay.TIRE_KEYS},
        track_name="Fuji Speedway!!",
        track_config=" Grand-Prix  ",
        car_path="Porsche 911 GT3 R (992)",
    )
    base.update(kw)
    return overlay.TelemetrySnapshot(**base)


def test_dataset_key_normalization():
    key = overlay.StintTracker.make_dataset_key(make_snap())
    assert key == "fuji_speedway+grand_prix+porsche_911_gt3_r_992"


def test_humidity_propagates_to_stint_end_payload():
    tracker = overlay.StintTracker()
    start = make_snap(session_time=0.0, lap=10, lap_dist_pct=0.1, on_pit_road=False, humidity=64.0)
    tracker.update(start)

    mid = make_snap(session_time=10.0, lap=10, lap_dist_pct=0.2, on_pit_road=False, humidity=66.0)
    tracker.update(mid)

    end = make_snap(
        session_time=20.0,
        lap=12,
        lap_dist_pct=0.0,
        on_pit_road=True,
        humidity=67.5,
        wear={t: 98.0 for t in overlay.TIRE_KEYS},
    )
    result = tracker.update(end)
    assert result is not None
    assert result["humidity"] == 67.5


def test_cumulative_lap_progress_energy_per_lap_calculation():
    worker = overlay.ModelWorker.__new__(overlay.ModelWorker)
    worker.stints = overlay.StintTracker()
    worker.stints.start_data = {"lap_progress": 25.25}
    worker.stints.current_energy = 500.0
    snap = make_snap(lap=27, lap_dist_pct=0.75)
    progress = worker._current_stint_laps_progress(snap)
    assert abs(progress - 2.5) < 1e-9
    energy_per_lap = worker.stints.current_energy / max(1.0, progress)
    assert abs(energy_per_lap - 200.0) < 1e-9


def test_debounced_save_and_final_flush():
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "model.json")
        storage = overlay.DataStorage(path)
        model = overlay.TireMLModel.__new__(overlay.TireMLModel)
        model.storage = storage
        model._dirty_keys = set()

        calls = []
        orig_save = storage.save

        def wrapped_save():
            calls.append(time.time())
            orig_save()

        storage.save = wrapped_save

        class DummyRLS:
            def __init__(self):
                self.n_updates = 0
                self.confidence = 0.0

            def update(self, _x, _y):
                self.n_updates += 1

            def to_dict(self):
                return {"n_updates": self.n_updates}

        model._rls = {t: DummyRLS() for t in overlay.TIRE_KEYS}

        sample = {
            "track_temp": 30.0,
            "air_temp": 20.0,
            "humidity": 55.0,
            "energy_per_lap": 100.0,
            "lf": 0.001,
            "rf": 0.001,
            "lr": 0.001,
            "rr": 0.001,
        }
        model.add_stint_sample("k", sample)
        assert len(calls) == 0
        model.flush_pending()
        assert len(calls) == 1


if __name__ == "__main__":
    test_dataset_key_normalization()
    test_humidity_propagates_to_stint_end_payload()
    test_cumulative_lap_progress_energy_per_lap_calculation()
    test_debounced_save_and_final_flush()
    print("ok")
