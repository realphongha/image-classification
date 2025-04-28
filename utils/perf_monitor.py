import random
from collections import deque, defaultdict


def mean(arr):
    assert len(arr)
    return sum(arr) / len(arr)


class PerfMonitorMixin:
    PERF_MONITOR = {}
    MAX_SIZE = 128
    # TYPE = "latest" => only calculate final latency based on latest `MAX_SIZE` updates
    # TYPE = "random" => calculate based on `MAX_SIZE` random samples in the whole run using Algorithm R
    TYPE = "random"
    CNT = {}  # count total for each name, only for TYPE = "random"

    def update_perf(self, name, latency, ignore_classwise=False):
        if not ignore_classwise:
            name = self.__class__.__name__ + "_" + name
        PerfMonitorMixin.update_perf_static(name, latency)

    @staticmethod
    def update_perf_static(name, latency):
        if name not in PerfMonitorMixin.PERF_MONITOR:
            if PerfMonitorMixin.TYPE == "latest":
                PerfMonitorMixin.PERF_MONITOR[name] = deque(
                    [], maxlen=PerfMonitorMixin.MAX_SIZE)
            elif PerfMonitorMixin.TYPE == "random":
                PerfMonitorMixin.PERF_MONITOR[name] = []
                PerfMonitorMixin.CNT[name] = 0
            else:
                raise NotImplementedError
        if PerfMonitorMixin.TYPE == "latest":
            PerfMonitorMixin.PERF_MONITOR[name].append(latency)
        elif PerfMonitorMixin.TYPE == "random":
            PerfMonitorMixin.CNT[name] += 1
            if len(PerfMonitorMixin.PERF_MONITOR[name]) < PerfMonitorMixin.MAX_SIZE:
                PerfMonitorMixin.PERF_MONITOR[name].append(latency)
            else:
                j = random.randrange(PerfMonitorMixin.CNT[name])
                if j < PerfMonitorMixin.MAX_SIZE:
                    PerfMonitorMixin.PERF_MONITOR[name][j] = latency

    def get_perf_for_this_class(self, name, display=True):
        name = self.__class__.__name__ + "_" + name
        return PerfMonitorMixin.get_perf(name, display)

    def get_all_perfs_for_this_class(self, display=True):
        class_name = self.__class__.__name__ + "_"
        all_perfs = {}
        perf_names = list(PerfMonitorMixin.PERF_MONITOR.keys())
        # Higest latency goes first
        perf_names.sort(key=lambda k: -mean(PerfMonitorMixin.PERF_MONITOR[k]))
        for name in perf_names:
            if name.startswith(class_name):
                all_perfs[name] = PerfMonitorMixin.get_perf(name, display)
        return all_perfs

    @staticmethod
    def get_perf(name, display=True):
        if name not in PerfMonitorMixin.PERF_MONITOR:
            if display:
                print(f"{name} is not in performance monitor!")
            return -1, -1
        latency = mean(PerfMonitorMixin.PERF_MONITOR[name])
        total_time = sum(PerfMonitorMixin.PERF_MONITOR[name])
        fps = 1 / latency
        if display:
            print("SPEED PERFORMANCE for '%s':" % name)
            print("FPS - %10.2f, Latency - %15.8f, Total time - %15.8f" %
                (fps, latency, total_time))
        return fps, latency

    @staticmethod
    def get_all_perfs(display=True):
        all_perfs = {}
        perf_names = list(PerfMonitorMixin.PERF_MONITOR.keys())
        # Slowest goes first
        perf_names.sort(key=lambda k: -mean(PerfMonitorMixin.PERF_MONITOR[k]))
        for name in perf_names:
            all_perfs[name] = PerfMonitorMixin.get_perf(name, display)
        return all_perfs


if __name__ == "__main__":
    PerfMonitorMixin.MAX_SIZE = 5
    PerfMonitorMixin.TYPE = "random"
    PerfMonitorMixin.update_perf_static("test", 0.1)
    PerfMonitorMixin.update_perf_static("test", 0.1)
    PerfMonitorMixin.update_perf_static("test", 0.1)
    PerfMonitorMixin.update_perf_static("test", 0.1)
    PerfMonitorMixin.update_perf_static("test", 0.1)
    PerfMonitorMixin.update_perf_static("test", 0.2)
    PerfMonitorMixin.update_perf_static("test", 0.3)
    PerfMonitorMixin.update_perf_static("test", 0.4)
    print(PerfMonitorMixin.PERF_MONITOR["test"])
    PerfMonitorMixin.get_all_perfs()
