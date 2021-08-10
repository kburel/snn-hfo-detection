import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Slider
import numpy as np
from brian2.units import second, ms
from snn_hfo_ieeg.user_facing_data import HfoDetectionRun, PlotMode
from snn_hfo_ieeg.plotting.persistence import save_or_show_channel_plot, should_save_plot, should_show_plot


class ChannelDebugError(Exception):
    def __init__(self, message, hfo_run: HfoDetectionRun):
        super().__init__(message)
        self.hfo_run = hfo_run


def plot_internal_channel_debug(hfo_run: HfoDetectionRun):
    raise ChannelDebugError(
        "plot_internal_channel_debug is just here for debugging purposes and should not be called",
        hfo_run)


def plot_raster(hfo_run: HfoDetectionRun):
    hfo_detection = hfo_run.detector.last_run
    if hfo_detection.result.total_amount == 0:
        return
    plt.plot(hfo_detection.analytics.spike_times*second/ms,
             hfo_run.analytics.neuron_ids, '.k')
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron index')
    save_or_show_channel_plot("raster", hfo_run)


def _plot_hfo_sample(hfo_run: HfoDetectionRun, start, stop, bandwidth_axes, spike_train_axes, raster_axes):
    analytics = hfo_run.detector.last_run.analytics
    signal_time = hfo_run.input.signal_time  # time from the detect_with_analytics
    # pattern signal in the original data set, retirved from detect_with_analytics
    signal_teacher = analytics.detections

    #-------------------%Set limits according to the mark%--------------------#

    #-------------------%Specify signal snippet to be plotted%--------------------------------------#

    indices_time = np.where((signal_time > start) & (signal_time < stop))

    # This is how I assume we can access the data:
    signal_r = np.array(analytics.filtered_spikes.ripple.signal)[
        indices_time] if analytics.filtered_spikes.ripple is not None else None
    signal_fr = np.array(analytics.filtered_spikes.fast_ripple.signal)[
        indices_time] if analytics.filtered_spikes.fast_ripple is not None else None
    signal_time = signal_time[indices_time]
    signal_teacher = np.array(signal_teacher)[indices_time]

    # ==========================================================================
    # GRID PLOT
    # ==========================================================================

    # =========================================================================
    # Plot Wideband, Ripple band and fr band signal
    # =========================================================================
    scale_fr = 6

    scale_ripple = 3
    shift_ripple = 1

    #-------------------%Plot fr band signal%--------------------------------#
    bandwidth_axes.plot(signal_time, signal_fr * scale_fr,
                        color='#8e5766', linewidth=1)

    ylim_up_fr = np.max(signal_fr) * scale_fr

    #-------------------%Shift up and plot Ripple band signal%---------------#
    bandwidth_axes.plot(signal_time, signal_r * scale_ripple +
                        shift_ripple * np.abs(np.min(signal_r*scale_ripple)) + ylim_up_fr, color='#8e5766', linewidth=1)

    ylim_up_r = np.max(signal_r) * scale_ripple + shift_ripple * \
        np.abs(np.min(signal_r*scale_ripple)) + ylim_up_fr

    #-------------------%Infdicate HFO marking with a line%------------------#
    bandwidth_axes.fill_between(signal_time, 2 * np.min(signal_fr) * scale_fr,
                                2.2 * np.min(signal_fr) * scale_fr, where=signal_teacher == 1,
                                facecolor='#595959', alpha=0.7, label='teacher')

    #-------------------%Set y limits of plot with signals%------------------#
    shift_y_lim_max = 1
    y_lim_max_signal = shift_y_lim_max * ylim_up_r
    y_lim_min_signal = 2.4 * np.min(signal_fr) * scale_fr
    bandwidth_axes.set_ylim((y_lim_min_signal,
                             y_lim_max_signal))

    # =========================================================================
    # Add amplitude scales and labels
    # =========================================================================
    x_line = 0.003
    x_text_uv = 0.01
    x_label = 0.005
    #----------------------------%ripple band signal%------------------------#
    reference_line_microvolts_ripple = 20
    bandwidth_axes.annotate("",
                            xy=(start - x_line,
                                signal_r[0] * scale_ripple +
                                shift_ripple * np.abs(np.min(signal_r*scale_ripple)) + ylim_up_fr -
                                reference_line_microvolts_ripple*scale_ripple/2),
                            xytext=(start - x_line,
                                    signal_r[0] * scale_ripple +
                                    shift_ripple * np.abs(np.min(signal_r*scale_ripple)) + ylim_up_fr +
                                    reference_line_microvolts_ripple*scale_ripple/2),
                            arrowprops=dict(arrowstyle='-'),
                            annotation_clip=False)

    bandwidth_axes.text(start - x_text_uv,
                        (signal_r[0] * scale_ripple +
                         shift_ripple * np.abs(np.min(signal_r*scale_ripple)) + ylim_up_fr),
                        r'%i $\mu$V' % reference_line_microvolts_ripple, verticalalignment='center',
                        rotation=0,
                        fontsize=10)

    ripple_label_position = 1.2 * \
        (np.mean(np.abs(signal_r[0:10])) * scale_ripple + shift_ripple *
         np.abs(np.min(signal_r*scale_ripple)) + ylim_up_fr)

    bandwidth_axes.text(start + x_label,
                        ripple_label_position,
                        'Ripple Band', verticalalignment='center',
                        fontsize=12)

    #-------------------%Fast ripple band signal%----------------------------#
    reference_line_microvolts_fr = 10
    bandwidth_axes.annotate("",
                            xy=(start - x_line,
                                signal_fr[0] * scale_fr - reference_line_microvolts_fr*scale_fr/2),
                            xytext=(start - x_line,
                                    signal_fr[0] * scale_fr + reference_line_microvolts_fr*scale_fr/2),
                            arrowprops=dict(arrowstyle='-'),
                            annotation_clip=False)

    bandwidth_axes.text(start - x_text_uv,
                        (signal_fr[0] * scale_fr),
                        r'%i $\mu$V' % reference_line_microvolts_fr, verticalalignment='center',
                        rotation=0,
                        fontsize=10)

    fr_label_position = 10 * (np.mean(np.abs(signal_fr[0:5])) * scale_fr)

    bandwidth_axes.text(start + x_label,
                        fr_label_position,
                        'Fast Ripple Band', verticalalignment='center',
                        fontsize=12)

    # ========================================================================================
    # Plot spikes
    # ========================================================================================

    # Previosuly I have the spikes stored in the dictionary, so I could loop over the keys, I am not sure something similar can be done now
    # In any case the Spikes are the ones returned by detect_with_analytics

    filtered_spikes = [analytics.filtered_spikes.fast_ripple.spike_trains.down,
                       analytics.filtered_spikes.fast_ripple.spike_trains.up,
                       analytics.filtered_spikes.ripple.spike_trains.down,
                       analytics.filtered_spikes.ripple.spike_trains.up]
    lineoffsets = 0.2
    for spikes in filtered_spikes:
        spikes_in_current_window = [spike for spike in spikes
                                    if start < spike < stop]

    #-------------------%Specify spikes%--------------------------------------#
        spike_train_axes.eventplot(spikes_in_current_window, color='#000000', linelengths=0.15,
                                   lineoffsets=lineoffsets, linewidth=1.5)
        lineoffsets += 0.2

    # Managing y labels for spike plots
    spike_train_axes.set_yticks(np.arange(0, 1, 0.2))

    labels = [item.get_text() for item in spike_train_axes.get_yticklabels()]
    labels[4] = 'R UP'
    labels[3] = 'R DN'
    labels[2] = 'FR UP'
    labels[1] = 'FR DN'
    spike_train_axes.set_yticklabels(labels, rotation=0, fontsize=10,
                                     verticalalignment='center')

    # =========================================================================
    # Raster plot
    # =========================================================================
    spikes_in_window = [(spike_index, spike_time) for spike_index, spike_time
                        in enumerate(analytics.spike_times)
                        if start < spike_time < stop]

    spike_times = [spike_time for _spike_index, spike_time in spikes_in_window]
    neuron_ids = [analytics.neuron_ids[spike_index]
                  for spike_index, _spike_time in spikes_in_window]
    raster_axes.plot(spike_times, neuron_ids,
                     '.', markersize=1.5,  color='#002699')
    raster_axes.yaxis.set_label_coords(-0.1, 0.5)
    raster_axes.set_xlabel('Time (ms)', fontsize=12)

    neuron_count = hfo_run.configuration.hidden_neuron_count
    # Managing y labels for spike plots
    raster_axes.set_yticks(
        np.arange(0, neuron_count, int(neuron_count / 5.0)))
    raster_axes.set_ylabel('Neuron ID', fontsize=12, x=- 0.01)

    # =========================================================================
    # Set limits
    # =========================================================================
    bandwidth_axes.set_xlim((start, stop))
    spike_train_axes.set_xlim((start, stop))
    raster_axes.set_xlim((start, stop))

    # =========================================================================
    # Extra figure settings
    # =========================================================================

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

    # =========================================================================
    # Managing x labels of time
    # =========================================================================
    spike_train_axes.tick_params(which='both', labelsize=10)

    raster_axes_labels = [np.round(tick, 2)
                          for tick in np.arange(start, stop, (stop-start)/5)]
    raster_axes.set_xticks(raster_axes_labels)


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
