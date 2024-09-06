# ThermalImagesDataset
Repository for scientific paper Multimodal driver condition monitoring system operating in the far infrared spectrum

# License
Use only for academic and/or research purposes. No commercial use.
Publication permitted only if the Data Sets are unmodified and subject to the same license terms.
Any publication must include a full citation to the paper in which the Data Sets were initially published by Knapik et al.

# Paper (Open Access)
https://www.mdpi.com/2079-9292/13/17/3502

Citation (BibTex):
```
@Article{electronics13173502,
AUTHOR = {Knapik, Mateusz and Cyganek, Bogusław and Balon, Tomasz},
TITLE = {Multimodal Driver Condition Monitoring System Operating in the Far-Infrared Spectrum},
JOURNAL = {Electronics},
VOLUME = {13},
YEAR = {2024},
NUMBER = {17},
ARTICLE-NUMBER = {3502},
URL = {https://www.mdpi.com/2079-9292/13/17/3502},
ISSN = {2079-9292},
ABSTRACT = {Monitoring the psychophysical conditions of drivers is crucial for ensuring road safety. However, achieving real-time monitoring within a vehicle presents significant challenges due to factors such as varying lighting conditions, vehicle vibrations, limited computational resources, data privacy concerns, and the inherent variability in driver behavior. Analyzing driver states using visible spectrum imaging is particularly challenging under low-light conditions, such as at night. Additionally, relying on a single behavioral indicator often fails to provide a comprehensive assessment of the driver’s condition. To address these challenges, we propose a system that operates exclusively in the far-infrared spectrum, enabling the detection of critical features such as yawning, head drooping, and head pose estimation regardless of the lighting scenario. It integrates a channel fusion module to assess the driver’s state more accurately and is underpinned by our custom-developed and annotated datasets, along with a modified deep neural network designed for facial feature detection in the thermal spectrum. Furthermore, we introduce two fusion modules for synthesizing detection events into a coherent assessment of the driver’s state: one based on a simple state machine and another that combines a modality encoder with a large language model. This latter approach allows for the generation of responses to queries beyond the system’s explicit training. Experimental evaluations demonstrate the system’s high accuracy in detecting and responding to signs of driver fatigue and distraction.},
DOI = {10.3390/electronics13173502}
}
```
# Sample videos
1. https://drive.google.com/file/d/10Evma4Jkmn5H_nu-rZJOxPJ8Lx4_i2m9/view?usp=drive_link
2. https://drive.google.com/file/d/1z7vNfgCQAbWVPxUZIRDGgk7pAdayjwvu/view?usp=drive_link
