class ChannelDebugError(Exception):
    def __init__(self, message, hfo_detection):
        super().__init__(message)
        self.hfo_detection = hfo_detection


def plot_internal_channel_debug(hfo_detection):
    raise ChannelDebugError(
        "plot_internal_channel_debug is just here for debugging purposes and should not be called",
        hfo_detection)
