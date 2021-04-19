
# HFO detection with a Spiking Neural Network 
**This code is still in development and as such welcomes suggestions.**
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
### **Input Parameters**: 
The parameters are read from two pre-created "**.mat**" files. 
The "ADM_parameters.mat" file contains the parameters used for the signal to spike converison phase of the SNN HFO detector. 
The "Network_parameters.mat" file contains the parameters used for the SNN architecture.
For more details regarding the architecture and SNN HFO detector see https://arxiv.org/abs/2009.11245

### **Input Data**: 
The data has to be structured in the following way:
```
iEEG_data = {}
iEEG_data['chb']: matrix of iEEG signal, each column is the iEEG signal from each recorded channel.
iEEG_data['t']: array containing the time vector of the recorded signal
```
The SNN HFO detector will run for all the channel from a single recorded interval and it will save the results in the dictionary "Test_Results".

The Filtered signal as well as the generated UP and DN spikes can be accessed trough the dictionaries "Signal"  and  "Spikes" respectively.


## How to use the code:
1. Install the spiking neural network simulator [brian2](https://brian2.readthedocs.io/en/stable/) as well as the toolbox for brian2 [teili](https://teili.readthedocs.io/en/latest/).
2. Make sure that the input data is in the correct format.
3. The rest of the code should run without a problem.

# Support

## This code has been written originally by:
* Karla Burelo
**kburel@ini.uzh.ch**



