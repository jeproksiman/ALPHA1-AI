from collections import deque, Counter
from time import monotonic


class PatternDetector:
    def __init__(self, min_repetitions=3, max_history=60, max_gap=300):
        self.history = deque(maxlen=max_history)
        self.min_repetitions = min_repetitions
        self.max_gap = max_gap
        self.suggested = set()

    def record(self, phrase):
        self.history.append((phrase, monotonic()))

    def suggestions(self):
        history = list(self.history)
        suggestions = []
        for size in (3, 2):
            counts = Counter()
            last_end = {}
            for start in range(len(history)-size+1):
                window = history[start:start+size]
                sequence = tuple(item[0] for item in window)
                if len(set(sequence)) < 2 or any(window[i+1][1]-window[i][1]>self.max_gap for i in range(size-1)):
                    continue
                if start <= last_end.get(sequence, -1):
                    continue
                counts[sequence] += 1
                last_end[sequence] = start+size-1
            for sequence, count in counts.items():
                if count >= self.min_repetitions and sequence not in self.suggested:
                    suggestions.append({'steps':list(sequence),'repetitions':count})
        return suggestions[:5]
