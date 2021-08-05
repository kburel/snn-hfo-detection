from statistics import mean
import numpy as np
import matplotlib.pyplot as plt


def _append_or_create(dict, key, value):
    if key not in dict:
        dict[key] = [value]
    else:
        dict[key].append(value)


def _convert_to_labels_to_hfo_rate_dict(intervals):
    label_to_hfo_rates = {}
    for hfo_detection_runs in intervals.values():
        for hfo_detection_run in hfo_detection_runs:
            _append_or_create(
                dict=label_to_hfo_rates,
                key=hfo_detection_run.metadata.channel_label,
                value=hfo_detection_run.hfo_detection.result.frequency * 60)

    return label_to_hfo_rates


def _plot_bar(axes, intervals):
    label_to_hfo_rates = _convert_to_labels_to_hfo_rate_dict(intervals)

    labels = list(label_to_hfo_rates.keys())
    hfo_rates = label_to_hfo_rates.values()

    mean_hfo_rates = [mean(_) for _ in hfo_rates]
    standard_deviations = [np.std(_) for _ in hfo_rates] \
        if len(intervals) > 1 else None
    axes.bar(
        x=labels,
        height=mean_hfo_rates,
        width=0.3,
        edgecolor='k',
        ecolor='#0218f5',
        alpha=0.9, color='#2f70b6',
        yerr=standard_deviations,
        capsize=2)


def _rotate_labels(axes):
    labels = axes.get_xticklabels()
    for label in labels:
        label.set_rotation(45)
        label.set_horizontalalignment('right')
        label.set_size = 16


def _set_layout(fig):
    plt.rc('font', family='sans-serif')
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25, left=0.1, wspace=0.2, hspace=0.2)


def _style_ticks(axes):
    axes.set_ylim(bottom=0)
    axes.set_xlabel('Electrode label', fontsize=18)
    axes.set_ylabel('HFO rate (event/min)', fontsize=18)

    axes.tick_params(axis='y', labelsize=16, length=8)
    axes.tick_params(axis='x', labelsize=16, length=8)


def _hide_spines(axes):
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)


def plot_mean_hfo_rate(intervals):
    if len(intervals) == 0:
        return

    fig, axes = plt.subplots(figsize=(15, 5))

    _set_layout(fig)
    _plot_bar(axes, intervals)
    _rotate_labels(axes)
    _style_ticks(axes)
    _hide_spines(axes)
