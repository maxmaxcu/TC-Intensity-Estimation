# Tropical-Cyclone-Intensity-Estimation
### Tropic Cyclone (TC) Intensity Estimation
Dataset of Tropical Cyclone for Image-to-intensity Regression (TCIR) [^TCIR] was put forward by Boyo Chen, BuoFu Chen and Hsuan-Tien Lin. Please browse web page [TCIR](https://www.csie.ntu.edu.tw/~htlin/program/TCIR/) for detail.
Single image in the TCIR dataset has the size of  201 (height) \* 201 (width) \* 4 (channels). Four channels are Infrared, Water vapor, Visible and Passive microwave, respectively. We just use Infrared and Passive microwave channels.

File TCIntensityEstimation has all the file about tropic cyclone intensity estimation problem, including source code, "how to get data" and a trianed model weights file.

- download source dataset (~13GB) and unzip

```
wget https://learner.csie.ntu.edu.tw/~boyochen/TCIR/TCIR-ALL_2017.h5.tar.gz
wget https://learner.csie.ntu.edu.tw/~boyochen/TCIR/TCIR-ATLN_EPAC_WPAC.h5.tar.gz
wget https://learner.csie.ntu.edu.tw/~boyochen/TCIR/TCIR-CPAC_IO_SH.h5.tar.gz
```
- preprocessing: 
 use convective_core_features.py to generate convective core features. Other preprocessing approaches were described in the paper.
 
- models:
  baseline models are modified to be compatible with the data shape used in our model. Adjusted loss funtion is in tools.py. The number of auxiliary features, adjusted loss coeffients and other parameters can be set with args.
 
