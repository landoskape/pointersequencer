import multiprocessing as mp
import threading


class PersistentPool:

    def __init__(self, workers=None):
        self._mp_pool_lock = threading.Lock()
        self._pool = None
        if workers is None:
            self._pool_count = min(2, int(mp.cpu_count() * 0.50))
        else:
            self._pool_count = workers
        assert self._pool_count > 0 and self._pool_count <= mp.cpu_count(), "Invalid number of workers"

    @property
    def mp_pool(self):
        with self._mp_pool_lock:
            if self._pool is None:
                self._pool = mp.Pool(self._pool_count)
        return self._pool

    def close(self):
        with self._mp_pool_lock:
            if self._pool:
                self._pool.close()
                self._pool = None

    def __del__(self):
        self.close()
