import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from brian2.units import second, ms
from snn_hfo_ieeg.user_facing_data import HfoDetectionRun


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
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')


def _plot_hfo_sample(hfo_run: HfoDetectionRun, start, stop):
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
    rows = 7
    columns = 6
    fig_height = 10
    length_x_axis = 5
    length_y_axis = 5

    height = length_y_axis * rows
    width = length_x_axis * columns

    plt.rc('font', family='sans-serif')

    plot_aspect_ratio = float(width)/float(height)
    fig = plt.figure(figsize=(fig_height * plot_aspect_ratio, fig_height))
    gs = gridspec.GridSpec(rows, columns,
                           width_ratios=[1]*6,
                           height_ratios=[1]*7
                           )

    axs0 = fig.add_subplot(gs[1:2, 1:])
    axs1 = fig.add_subplot(gs[2:3, 1:])
    axs2 = fig.add_subplot(gs[3:4, 1:])

    # =========================================================================
    # Plot Wideband, Ripple band and fr band signal
    # =========================================================================
    scale_fr = 6

    scale_ripple = 3
    shift_ripple = 1

    #-------------------%Plot fr band signal%--------------------------------#
    axs0.plot(signal_time, signal_fr * scale_fr, color='#8e5766', linewidth=1)

    ylim_up_fr = np.max(signal_fr) * scale_fr

    #-------------------%Shift up and plot Ripple band signal%---------------#
    axs0.plot(signal_time, signal_r * scale_ripple +
              shift_ripple * np.abs(np.min(signal_r*scale_ripple)) + ylim_up_fr, color='#8e5766', linewidth=1)

    ylim_up_r = np.max(signal_r) * scale_ripple + shift_ripple * \
        np.abs(np.min(signal_r*scale_ripple)) + ylim_up_fr

    #-------------------%Infdicate HFO marking with a line%------------------#
    axs0.fill_between(signal_time, 2 * np.min(signal_fr) * scale_fr,
                      2.2 * np.min(signal_fr) * scale_fr, where=signal_teacher == 1,
                      facecolor='#595959', alpha=0.7, label='teacher')

    #-------------------%Set y limits of plot with signals%------------------#
    shift_y_lim_max = 1
    y_lim_max_signal = shift_y_lim_max * ylim_up_r
    y_lim_min_signal = 2.4 * np.min(signal_fr) * scale_fr
    axs0.set_ylim((y_lim_min_signal,
                   y_lim_max_signal))

    # =========================================================================
    # Add amplitude scales and labels
    # =========================================================================
    x_line = 0.003
    x_text_uv = 0.017
    x_label = 0.005
    #----------------------------%ripple band signal%------------------------#
    reference_line_microvolts_ripple = 20
    axs0.annotate("",
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

    axs0.text(start - x_text_uv,
              (signal_r[0] * scale_ripple +
               shift_ripple * np.abs(np.min(signal_r*scale_ripple)) + ylim_up_fr),
              r'%i $\mu$V' % reference_line_microvolts_ripple, verticalalignment='center',
              rotation=0,
              fontsize=10)

    ripple_label_position = 1.2 * \
        (np.mean(np.abs(signal_r[0:10])) * scale_ripple + shift_ripple *
         np.abs(np.min(signal_r*scale_ripple)) + ylim_up_fr)

    axs0.text(start + x_label,
              ripple_label_position,
              'Ripple Band', verticalalignment='center',
              fontsize=12)

    #-------------------%Fast ripple band signal%----------------------------#
    reference_line_microvolts_fr = 10
    axs0.annotate("",
                  xy=(start - x_line,
                      signal_fr[0] * scale_fr - reference_line_microvolts_fr*scale_fr/2),
                  xytext=(start - x_line,
                          signal_fr[0] * scale_fr + reference_line_microvolts_fr*scale_fr/2),
                  arrowprops=dict(arrowstyle='-'),
                  annotation_clip=False)

    axs0.text(start - x_text_uv,
              (signal_fr[0] * scale_fr),
              r'%i $\mu$V' % reference_line_microvolts_fr, verticalalignment='center',
              rotation=0,
              fontsize=10)

    fr_label_position = 10 * (np.mean(np.abs(signal_fr[0:5])) * scale_fr)

    axs0.text(start + x_label,
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
        axs1.eventplot(spikes_in_current_window, color='#000000', linelengths=0.15,
                       lineoffsets=lineoffsets, linewidth=1.5)
        lineoffsets += 0.2

    # Managing y labels for spike plots
    axs1.set_yticks(np.arange(0, 1, 0.2))

    labels = [item.get_text() for item in axs1.get_yticklabels()]
    labels[4] = 'R UP'
    labels[3] = 'R DN'
    labels[2] = 'FR UP'
    labels[1] = 'FR DN'
    axs1.set_yticklabels(labels, rotation=0, fontsize=10,
                         verticalalignment='center')

    # =========================================================================
    # Raster plot
    # =========================================================================
    axs2.plot(analytics.spike_times, analytics.neuron_ids,
              '.', markersize=1.5,  color='#002699')
    axs2.yaxis.set_label_coords(-0.1, 0.5)
    axs2.set_xlabel('Time (ms)', fontsize=12)

    neuron_count = hfo_run.configuration.hidden_neuron_count
    # Managing y labels for spike plots
    axs2.set_yticks(
        np.arange(0, neuron_count, int(neuron_count / 5.0)))
    axs2.set_ylabel('Neuron ID', fontsize=12, x=- 0.01)

    # =========================================================================
    # Set limits
    # =========================================================================
    axs0.set_xlim((start, stop))
    axs1.set_xlim((start, stop))
    axs2.set_xlim((start, stop))

    # =========================================================================
    # Extra figure settings
    # =========================================================================

    axs0.tick_params(which='both', bottom=False, top=False,
                     left=False, labelbottom=False, labelleft=False)
    axs1.tick_params(which='both', bottom=False, top=False,
                     labelbottom=False, left=True)
    axs2.tick_params(which='both', left=True, labelleft=False)

    axs0.spines['top'].set_visible(False)
    axs0.spines['right'].set_visible(False)
    axs0.spines['left'].set_visible(False)
    axs0.spines['bottom'].set_visible(False)

    axs1.spines['top'].set_visible(False)
    axs1.spines['right'].set_visible(False)
    axs1.spines['bottom'].set_visible(False)

    axs2.spines['top'].set_visible(False)
    axs2.spines['right'].set_visible(False)

    # =========================================================================
    # Managing x labels of time
    # =========================================================================
    axs1.tick_params(which='both', labelsize=10)

    divider = 30
    axs2.set_xticks(np.arange(start, stop, divider*1e-3))
    axs0.set_xticks(np.arange(start, stop, divider*1e-3))

    labels = [item.get_text() for item in axs2.get_xticklabels()]
    for i in range(1, axs2.get_xticks().size, 1):
        labels[i] = '%0.2d' % ((start*1e3 + i*(divider)) - start*1e3)

    axs2.set_xticklabels(labels, rotation=0, fontsize=10,
                         verticalalignment='top')


def plot_hfo_samples(hfo_detection_run: HfoDetectionRun):
    periods = hfo_detection_run.detector.last_run.analytics.periods
    for start, stop in zip(periods.start, periods.stop):
        _plot_hfo_sample(hfo_detection_run,
                         np.float64(start), np.float64(stop))
