from heapq import heappush, heappop, heapify

class UpdateablePriorityQueue():
    def __init__(self, initial_values=None):
        if initial_values is None:
            self._heap = []
            self._dict = {}
        elif isinstance(initial_values, list):
            self._heap = initial_values.copy()
            self._dict = {key: priority for (priority, key) in initial_values}
        elif isinstance(initial_values, dict):
            self._dict = initial_values.copy()
            self._heap = [(priority, key) for (key, priority) in initial_values.items()]
        else:
            raise TypeError("initial_values must be a list or dict, not {}".format(type(initial_values).__name__))

        heapify(self._heap)

    def _remove_obsolete_entries(self):
        """
        Removes obsolete entries from heap
        """
        priority, key = self._heap[0]
        while (key not in self._dict) or (self._dict[key] != priority):
            heappop(self._heap)
            if not self._heap:
                break
            priority, key = self._heap[0]

    def pop(self):
        if not self:
            raise IndexError("Queue is empty")

        self._remove_obsolete_entries()

        priority, key = heappop(self._heap)
        del self._dict[key]

        return priority, key

    def peek(self):
        if not self:
            raise IndexError("Queue is empty")
        self._remove_obsolete_entries()
        priority, key = self._heap[0]
        return key, priority

    def __contains__(self, key):
        return key in self._dict

    def __getitem__(self, key):
        return self._dict[key]

    def __len__(self):
        return len(self._dict)

    def __setitem__(self, key, priority):
        self.push(key, priority)

    def __iter__(self):
        return iter(self._dict)

    def push(self, key, priority):
        self._dict[key] = priority
        heappush(self._heap, (priority, key))
