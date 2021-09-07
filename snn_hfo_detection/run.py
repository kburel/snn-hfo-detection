from snn_hfo_detection.entrypoint.hfo_detection import run_hfo_detection_with_configuration
from snn_hfo_detection.entrypoint.argument_parsing import parse_arguments, convert_arguments_to_config, convert_arguments_to_custom_overrides


def run_hfo_detection(hfo_cb):
    arguments = parse_arguments()
    configuration = convert_arguments_to_config(arguments)
    custom_overrides = convert_arguments_to_custom_overrides(arguments)
    run_hfo_detection_with_configuration(
        configuration=configuration,
        custom_overrides=custom_overrides,
        hfo_cb=hfo_cb)
