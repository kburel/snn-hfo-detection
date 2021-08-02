class TotalDebugError(Exception):
    def __init__(self, message, hfo_detections):
        super().__init__(message)
        self.hfo_detections = hfo_detections


def plot_internal_total_debug(hfo_detections):
    raise TotalDebugError(
        "plot_internal_total_debug is just here for debugging purposes and should not be called",
        hfo_detections)
