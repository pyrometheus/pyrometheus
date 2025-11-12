from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import time
from typing import Dict, Optional, Callable
from contextlib import contextmanager


@dataclass
class SerialTimer:
    """
    Minimal high-resolution timer for a single-process NumPy codebase.
    Usage:
        T = SerialTimer()
        t0 = T.start()
        # ... work ...
        T.record('rhs_eval', T.stop(t0))
        T.report(total='outer_time_loop')  # optional: set which key is 'total'
    """
    sums: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    calls: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # --- basic API ---
    def start(self) -> float:
        return time.perf_counter()

    def stop(self, t0: float) -> float:
        return time.perf_counter() - t0

    def record(self, name: str, dt: float) -> None:
        self.sums[name] += dt
        self.calls[name] += 1

    # --- convenience: context manager ---
    @contextmanager
    def timeit(self, name: str):
        t0 = self.start()
        try:
            yield
        finally:
            self.record(name, self.stop(t0))

    # --- convenience: decorator ---
    def wrap(self, name: str) -> Callable:
        def _decorator(fn: Callable) -> Callable:
            def _wrapped(*args, **kwargs):
                t0 = self.start()
                try:
                    return fn(*args, **kwargs)
                finally:
                    self.record(name, self.stop(t0))
            return _wrapped
        return _decorator

    # --- reporting ---
    def report(self, total: Optional[str] = None,
               sort_by: str = "time") -> None:
        """
        Print a table of timings.
        total: if provided, percentages are computed vs sums[total]; otherwise
        vs sum of all.
        sort_by: 'time' (default) or 'name'.
        """
        rows = []
        grand_total = self.sums.get(total, None)
        if grand_total is None:
            grand_total = sum(self.sums.values()) or 1.0

        for k in self.sums:
            avg = self.sums[k] / max(self.calls[k], 1)
            pct = 100.0 * (
                self.sums[k] / grand_total if grand_total > 0 else 0.0
            )
            rows.append((k, self.calls[k], self.sums[k], avg, pct))

        if sort_by == "name":
            rows.sort(key=lambda r: r[0])
        else:
            rows.sort(key=lambda r: r[2], reverse=True)

        print("\n=== Serial Timings ===")
        print(f"{'name':26s} {'calls':>6s} {'total(s)':>10s} {'avg(s)':>10s} "
              f"{'%':>7s}")
        for k, c, tot, avg, pct in rows:
            print(f"{k:26s} {c:6d} {tot:10.6f} {avg:10.6f} {pct:7.1f}")
        if total:
            print(f"(Reference total: '{total}' = {grand_total:.6f} s)")
        else:
            print(f"(Reference total: sum of rows = {grand_total:.6f} s)")

    # --- utilities ---
    def reset(self) -> None:
        self.sums.clear()
        self.calls.clear()
