import time

class ScopedTimer:
    def __init__(self, name):
        self.name = name
        self.start = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dt = (time.perf_counter() - self.start) * 1000
        # Выводим, только если дольше 1мс, чтобы не спамить
        if dt > 1.0:
            print(f"[TIME] {self.name}: {dt:.2f} ms")