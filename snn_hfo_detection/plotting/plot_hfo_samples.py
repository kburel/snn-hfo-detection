import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Slider
import numpy as np
from snn_hfo_detection.user_facing_data import HfoDetectionRun
from snn_hfo_detection.plotting.persistence import save_or_show_channel_plot, should_save_plot, should_show_plot


def _get_start_to_stop_indices(times: np.array, start, stop):
    start_index = np.searchsorted(times, start)
    end_index = np.searchsorted(times, stop)
    return start_index, end_index


def _should_draw_ripple(hfo_run):
    analytics = hfo_run.detector.last_run.analytics
    return analytics.filtered_spikes.ripple is not None and analytics.filtered_spikes.ripple != 'None'


def _should_draw_fast_ripple(hfo_run):
    analytics = hfo_run.detector.last_run.analytics
    return analytics.filtered_spikes.fast_ripple is not None and analytics.filtered_spikes.fast_ripple != 'None'


def _plot_bandwidth(bandwidth_axes, hfo_run, start, stop):
    analytics = hfo_run.detector.last_run.analytics
    start_index, stop_index = _get_start_to_stop_indices(
        hfo_run.input.signal_time, start, stop)

    should_draw_ripple = _should_draw_ripple(hfo_run)
    should_draw_fast_ripple = _should_draw_fast_ripple(hfo_run)

    signal_r = np.array(analytics.filtered_spikes.ripple.signal[
        start_index: stop_index]) if should_draw_ripple else np.zeros(stop_index - start_index)
    signal_fr = np.array(analytics.filtered_spikes.fast_ripple.signal[
        start_index: stop_index]) if should_draw_fast_ripple else np.zeros(stop_index - start_index)
    signal_time = hfo_run.input.signal_time[start_index: stop_index]

    scale_fr = 6
    scale_ripple = 3
    shift_ripple = 1

    if should_draw_fast_ripple:
        bandwidth_axes.plot(signal_time, signal_fr * scale_fr,
                            color='#8e5766', linewidth=1)

    ylim_up_fr = max(np.max(signal_r), np.max(signal_fr)) * scale_fr

    if should_draw_ripple:
        bandwidth_axes.plot(signal_time, signal_r * scale_ripple +
                            shift_ripple * np.abs(np.min(signal_r*scale_ripple)) + ylim_up_fr, color='#8e5766', linewidth=1)

    ylim_up_r = np.max(signal_r) * scale_ripple + shift_ripple * \
        np.abs(np.min(signal_r*scale_ripple)) + ylim_up_fr

    signal_teacher = np.array(
        analytics.detections[start_index: stop_index])
    bandwidth_axes.fill_between(signal_time, 2 * np.min(signal_fr) * scale_fr,
                                2.2 * np.min(signal_fr) * scale_fr, where=signal_teacher == 1,
                                facecolor='#595959', alpha=0.7, label='teacher')

    shift_y_lim_max = 1
    y_lim_max_signal = shift_y_lim_max * ylim_up_r
    y_lim_min_signal = 2.4 * np.min(signal_fr) * scale_fr
    bandwidth_axes.set_ylim((y_lim_min_signal,
                            y_lim_max_signal))

    _add_labels(bandwidth_axes, start, stop, hfo_run)


def _add_labels(bandwidth_axes, start, stop, hfo_run):
    x_line = 0.003
    x_text_uv = 0.01
    x_label = 0.005
    reference_line_microvolts_ripple = 20
    r_base_y = 100
    y_offset = 30

    should_draw_ripple = _should_draw_ripple(hfo_run)
    should_draw_fast_ripple = _should_draw_fast_ripple(hfo_run)

    if should_draw_ripple:
        bandwidth_axes.annotate("",
                                xy=(start - x_line,
                                    r_base_y - y_offset),
                                xytext=(start - x_line,
                                        r_base_y + y_offset),
                                arrowprops=dict(arrowstyle='-'),
                                annotation_clip=False)

        bandwidth_axes.text(start - x_text_uv,
                            (r_base_y),
                            rf'{reference_line_microvolts_ripple} $\mu$V', verticalalignment='center',
                            rotation=0,
                            fontsize=10)

        ripple_label_position = 100

        bandwidth_axes.text(start + x_label,
                            ripple_label_position,
                            'Ripple Band', verticalalignment='center',
                            fontsize=12)

    if should_draw_fast_ripple:
        reference_line_microvolts_fr = 10
        fr_base_y = 0
        bandwidth_axes.annotate("",
                                xy=(start - x_line,
                                    fr_base_y - y_offset),
                                xytext=(start - x_line,
                                        fr_base_y + y_offset),
                                arrowprops=dict(arrowstyle='-'),
                                annotation_clip=False)

        bandwidth_axes.text(start - x_text_uv,
                            (fr_base_y),
                            fr'{reference_line_microvolts_fr} $\mu$V', verticalalignment='center',
                            rotation=0,
                            fontsize=10)

        fr_label_position = 30
        bandwidth_axes.text(start + x_label,
                            fr_label_position,
                            'Fast Ripple Band', verticalalignment='center',
                            fontsize=12)

    bandwidth_axes.set_xlim((start, stop))


def _plot_spike_trains(spike_train_axes, hfo_run, start, stop):
    analytics = hfo_run.detector.last_run.analytics
    should_draw_ripple = _should_draw_ripple(hfo_run)
    should_draw_fast_ripple = _should_draw_fast_ripple(hfo_run)

    filtered_spikes = []
    if should_draw_ripple:
        filtered_spikes.append(
            analytics.filtered_spikes.ripple.spike_trains.up)
        filtered_spikes.append(
            analytics.filtered_spikes.ripple.spike_trains.down)

    if should_draw_fast_ripple:
        filtered_spikes.append(
            analytics.filtered_spikes.fast_ripple.spike_trains.up)
        filtered_spikes.append(
            analytics.filtered_spikes.fast_ripple.spike_trains.down)

    lineoffsets = 0.2
    for spikes in filtered_spikes:
        start_index, stop_index = _get_start_to_stop_indices(
            spikes, start, stop)
        spikes_in_current_window = spikes[start_index: stop_index]

        spike_train_axes.eventplot(spikes_in_current_window, color='#000000', linelengths=0.15,
                                   lineoffsets=lineoffsets, linewidth=1.5)
        lineoffsets += 0.2

    spike_train_axes.set_yticks(np.arange(0, 1, 0.2))

    labels = [item.get_text() for item in spike_train_axes.get_yticklabels()]
    labels[4] = 'R UP'
    labels[3] = 'R DN'
    labels[2] = 'FR UP'
    labels[1] = 'FR DN'
    spike_train_axes.set_yticklabels(labels, rotation=0, fontsize=10,
                                     verticalalignment='center')
    spike_train_axes.set_xlim((start, stop))


def _plot_raster(raster_axes, hfo_run, start, stop):
    analytics = hfo_run.detector.last_run.analytics
    start_index, stop_index = _get_start_to_stop_indices(
        analytics.spike_times, start, stop)
    spike_times = analytics.spike_times[start_index:stop_index]
    neuron_ids = analytics.neuron_ids[start_index:stop_index]
    raster_axes.plot(spike_times, neuron_ids,
                     '.', markersize=1.5,  color='#002699')
    raster_axes.yaxis.set_label_coords(-0.1, 0.5)
    raster_axes.set_xlabel('Time (s)', fontsize=12)

    neuron_count = hfo_run.configuration.hidden_neuron_count

    raster_axes.set_yticks(
        np.arange(0, neuron_count, int(neuron_count / 5.0)))
    raster_axes.set_ylabel('Neuron ID', fontsize=12, x=- 0.01)

    raster_axes_labels = [np.round(tick, 2)
                          for tick in np.arange(start, stop, (stop-start)/5)]
    raster_axes.set_xticks(raster_axes_labels)
    raster_axes.set_xlim((start, stop))


def _hide_ticks_and_spines(bandwidth_axes, spike_train_axes, raster_axes):
    bandwidth_axes.tick_params(which='both', bottom=False, top=False,
                               left=False, labelbottom=False, labelleft=False)
    spike_train_axes.tick_params(which='both', bottom=False, top=False,
                                 labelbottom=False, left=True)
    raster_axes.tick_params(which='both', left=True, labelleft=False)

    bandwidth_axes.spines['top'].set_visible(False)
    bandwidth_axes.spines['right'].set_visible(False)
    bandwidth_axes.spines['left'].set_visible(False)
    bandwidth_axes.spines['bottom'].set_visible(False)

    spike_train_axes.spines['top'].set_visible(False)
    spike_train_axes.spines['right'].set_visible(False)
    spike_train_axes.spines['bottom'].set_visible(False)

    raster_axes.spines['top'].set_visible(False)
    raster_axes.spines['right'].set_visible(False)


def _plot_hfo_sample(hfo_run: HfoDetectionRun, start, stop, bandwidth_axes, spike_train_axes, raster_axes):
    _plot_bandwidth(bandwidth_axes=bandwidth_axes,
                    hfo_run=hfo_run,
                    start=start,
                    stop=stop)
    _plot_spike_trains(spike_train_axes=spike_train_axes,
                       hfo_run=hfo_run,
                       start=start,
                       stop=stop)
    _plot_raster(raster_axes=raster_axes,
                 hfo_run=hfo_run,
                 start=start,
                 stop=stop)

    _hide_ticks_and_spines(bandwidth_axes=bandwidth_axes,
                           spike_train_axes=spike_train_axes,
                           raster_axes=raster_axes)


def plot_hfo_samples(hfo_detection_run: HfoDetectionRun):
    periods = hfo_detection_run.detector.last_run.analytics.periods
    fig_height = 6
    fig_width = 10
    rows = 4
    columns = 1
    fig = plt.figure(figsize=(fig_width, fig_height))

    plt.rc('font', family='sans-serif')

    spec = gridspec.GridSpec(rows, columns,
                             figure=fig,
                             hspace=0.7)

    bandwidth_axes = fig.add_subplot(spec[0, 0])
    spike_train_axes = fig.add_subplot(spec[1, 0])
    raster_axes = fig.add_subplot(spec[2, 0])
    slider_axes = fig.add_subplot(8, 1, 8)

    period_windows = list(zip(periods.start, periods.stop))
    if len(period_windows) == 0:
        return

    slider = Slider(slider_axes,
                    'Period Index\n(Interactive)',
                    1,
                    len(period_windows) + 1,
                    valinit=1,
                    valstep=1.0)

    initial_start, initial_stop = period_windows[0]
    _plot_hfo_sample(hfo_detection_run,
                     np.float64(initial_start), np.float64(initial_stop),
                     bandwidth_axes, spike_train_axes, raster_axes)

    def plot_time(one_based_index):
        start, stop = period_windows[int(np.round(one_based_index - 1))]
        _plot_hfo_sample(hfo_detection_run,
                         np.float64(start), np.float64(stop),
                         bandwidth_axes, spike_train_axes, raster_axes)
        fig.canvas.draw_idle()
    slider.on_changed(plot_time)

    if should_show_plot(hfo_detection_run.configuration):
        plt.show()
    if should_save_plot(hfo_detection_run.configuration):
        for one_based_index in range(1, len(period_windows) + 1):
            slider.set_val(one_based_index)
            save_or_show_channel_plot(
                f'hfo_sample_period_{one_based_index}', hfo_detection_run)
