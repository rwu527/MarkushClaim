
from threading import Lock

class Counter:
    def __init__(self):
        self._value = 1
        self._lock = Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
    
    @property
    def value(self):
        with self._lock:
            return self._value
        
global_r_group_counter = Counter()
global_x_group_counter = Counter()
global_z_group_counter = Counter()