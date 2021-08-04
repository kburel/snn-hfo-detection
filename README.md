
# HFO detection with a Spiking Neural Network 
**This code is still in development and as such welcomes suggestions.**

[![Continuous Integration](https://github.com/kburel/SNN_HFO_iEEG/actions/workflows/continuous-integration.yml/badge.svg)](https://github.com/kburel/SNN_HFO_iEEG/actions/workflows/continuous-integration.yml)
[![DOI](https://zenodo.org/badge/359535894.svg)](https://zenodo.org/badge/latestdoi/359535894)
---
This code is intended to be used for the detection of High Frequency Oscillations (HFO) in the following data:
* Electroencephalogram EEG 
* Electrocorticography ECoG 
* Intracranial Electroencephalogram iEEG 

## Introduction
### HFO working definition
HFO are recognized as biomarkers for epileptogenic brain tissue. HFOs are generally viewed as spontaneous EEG patterns in the frequency range between 80 to 500 Hz that consist of at least four oscillations that clearly stand out of the background activity. [HFO Review](https://doi.org/10.1016/j.clinph.2019.01.016)   


### Uses of HFO
Interictal HFOs have proven more specific in localizing the seizure onset zone (SOZ) than spikes and have presented a good association with the post-surgery outcome in epilepsy patients. We thus validated the clinical relevance of the HFO area in the individual patient with an automated procedure. This is a prerequisite before HFOs can guide surgical treatment in multi-center studies.

### Research Papers

<a id="1">[1]</a> Burnos S., Hilfiker P., Surucu O., Scholkmann F., Krayenbühl N., Grunwald T. Sarnthein J. Human intracranial high frequency oscillations (HFOs) detected by automatic time-frequency analysis. PLoS One 9, e94381,  [doi:10.1371/journal.pone.0094381]( https://www.doi.org/10.1371/journal.pone.0094381)	 (2014). 

<a id="2">[2]</a> Burnos S., Frauscher B., Zelmann R., Haegelen C., Sarnthein J., Gotman J. The morphology of high frequency oscillations (HFO) does not improve delineating the epileptogenic zone. Clin Neurophysiol 127, 2140-2148, [doi:10.1016/j.clinph.2016.01.002]( https://www.doi.org/10.1016/j.clinph.2016.01.002) (2016).


<a id="3">[3]</a> Fedele T., van 't Klooster M., Burnos S., Zweiphenning W., van Klink N., Leijten F., Zijlmans M., Sarnthein J. Automatic detection of high frequency oscillations during epilepsy surgery predicts seizure outcome. Clin Neurophysiol 127, 3066-3074, [doi:10.1016/j.clinph.2016.06.009 ](  https://www.sciencedirect.com/science/article/pii/S1388245716304394?via%3Dihub) (2016).

<a id="4">[4]</a> Fedele T., Burnos S., Boran E., Krayenbühl N., Hilfiker P., Grunwald T. and Sarnthein J. Resection of high frequency oscillations predicts seizure outcome in the individual patient. Sci Rep 7, 13836,  [doi:10.1038/s41598-017-13064-1]( https://www.doi.org/10.1038/s41598-017-13064-1) (2017).

<a id="5">[5]</a> Fedele T., Ramantani G., Burnos S., Hilfiker P., Curio G., Grunwald T., Krayenbühl N., Sarnthein J. Prediction of seizure outcome improved by fast ripples detected in low-noise intraoperative corticogram. Clin Neurophysiol 128, 1220-1226, [doi:10.1016/j.clinph.2017.03.038]( https://www.doi.org/10.1016/j.clinph.2017.03.038) (2017).


<a id="6">[6]</a> Boran E., Ramantani G., Krayenbühl N., Schreiber M., König K., Fedele T. and Sarnthein J. High-density ECoG improves the detection of high frequency oscillations that predict seizure outcome. Clin Neurophysiol 130, 1882-1888, [doi:10.1016/j.clinph.2019.07.008]( https://www.doi.org/10.1016/j.clinph.2019.07.008) (2019).

<a id="7">[7]</a> Boran E., Sarnthein J., Krayenbühl N., Ramantani G. and Fedele T. High-frequency oscillations in scalp EEG mirror seizure frequency in pediatric focal epilepsy. Sci Rep 9, 16560, [doi:10.1038/s41598-019-52700-w]( https://www.doi.org/10.1038/s41598-019-52700-w) (2019).

<a id="7">[8]</a> Burelo K., Sharifshazileh M., Krayenbühl N., Ramantani G., Indiveri G., and Sarnthein J. A spiking neural network (SNN) for detecting high frequency oscillations (HFOs) in the intraoperative ECoG. Sci Rep 11, 6719, [doi:10.1038/s41598-021-85827-w]( https://doi.org/10.1038/s41598-021-85827-w) (2021).


<a id="7">[9]</a> Sharifshazileh M., Burelo K., Sarnthein J., and  Indiveri G. An electronic neuromorphic system for real-time detection of high
frequency oscillations (HFOs) in intracranial EEG. NatCom (accepted) (2021)

## How the detector works:
For more details regarding the architecture and SNN HFO detector see https://arxiv.org/abs/2009.11245

## Input data format: 
Each file containing the interval data must be a matlab file with the following variables:
- `times`: array containing the times of the recorded signal in seconds
- `channels`: matrix of iEEG signal. Each row is a channel, each column the signal at the time of the corresponding index
- `channel_labels`: character matrix of the channels' names. Fill it like `channel_labels = ['name_one'; 'name_two'; 'name_three']`.

You can see valid example files at [tests/integration/data](https://github.com/kburel/snn-hfo-detection/tree/main/tests/integration/data)
## Instructions
This project uses [poetry](https://python-poetry.org/) to manage its dependencies. You can download it via
```bash
pip install --user poetry
```
then clone this repository, `cd` into it and run
```bash
poetry install
```
Place your data in the folder `data/` in the form of `I<interval>.mat`, e.g.:
```bash
SNN_HFO_iEEG/data/I1.mat
```
then run the code via
```bash
poetry run ./run.py <mode>
```
where `<mode>` is one of either `ieeg`, `ecog` or `scalp`.
If you run into problems, you can always run
```bash
poetry run ./run.py --help
``` 
And if you're still stuck, feel free to open an [issue](https://github.com/kburel/SNN_HFO_iEEG/issues/new) and we will help.

## Usage Examples
Show help:
```bash
poetry run ./run.py --help
```

When running, you need to specify how the data was obtained in order to run the right analyzers. We support the following modes:
- **ieeg**: Data was obtained via iEEG, the ripple bandwidth (80-250 Hz) and the fast ripple bandwidth (250-500 Hz) will be analyzed
- **ecog**: Data was obtained via eCoG, only the fast ripple bandwidth will be analyzed
- **scalp**: Data was obtained over the scalp via EEG, only the ripple bandwidth will be analyzed

Analyze all available data in iEEG mode:
```bash
poetry run ieeg ./run.py
```

Run in iEEG mode with custom data path:
```bash
poetry run ./run.py ieeg --data-path path/to/data
```

Analyze all available data in iEEG mode with an SNN with 100 hidden neurons:
```bash
poetry run ieeg ./run.py --hidden-neurons 100
```

 
Only analyze channels 2, 3 and 5 in eCoG mode:
```bash
# Since the channels are imported from matlab, they are 1 based
poetry run ecog ./run.py ecog --channels 2 3 5
```

Only analyze the first 100 seconds of the datasets in scalp mode:
```bash
poetry run scalp ./run.py scalp --duration 100
```

Only analyze the intervals 2, 3, 4, 6, 7 and 8 in iEEG mode:
```bash
poetry run ./run.py ieeg --intervals 2 3 4 6 7 
```

All options can be freely combined. For example, the following will construct an SNN with 256 neurons and
analyze the intervals 3 and 4 of in the channels 1 and 2
while only looking at the first 300 seconds in iEEG mode for data in ./ieeg-data:
```bash
poetry run ./run.py iieg --data-path ./ieeg-data --hidden-neurons 256 --intervals 3 4 --channels 1 2 --duration 300
```

## Plotting
The output can be plotting during the run in various ways by using `--plot`. The specified plots are created either after every channel
or after the entire patient. Note that multiple plots can be speficied.

### Per channel plots
- **raster**: Classic neuron ID to spike time raster plot. On gets drawn when an HFO was detected.

### Per patient plots

## This code has been written originally by:
* Karla Burelo
**kburel@ini.uzh.ch**



