# **Masked Autoencoder SHM**
#### Author: Yhorman Bedoya

This repository contains the code to replicate the results of the master thesis Enhancing Structural Health Monitoring through Self-supervised Learning: An Application of Masked Autoencoders on Anomaly Detection and Traffic Load Estimation.

Here we will describe how to launch the masked autoencoder on both applications Anomaly detection and Traffic load estimation.

## **Prerequisites**

To use this application, you will need to have a virtual environment with Python 3.7.

## **Installation and setting**

The necessary libraries are available in the file requirements.txt, to install it, simply run the following command:

```
pip install -r requirements.txt
```

Then go to the file /.../timm/models/layers/helpers.py in your virtual environment, and change import libraries for the following lines

```
import torch
from itertools import repeat
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs
```

## **Usage**
### **Pre-training**

For both applications the pre-training phase is common, the only difference is the dataloader.
For anomaly detection, in the file main_pretrain.py import the dataloader as follows:

```
from util.DataLoadersINSIST.SHM_DataSet import SHMDataset
```

For TLE, in the file main_pretrain.py import the dataloader as follows:

```
from util.DataLoadersSacertisLabels.SHM_DataSet import SHMDataset
```

For TLE the dataloader call the following files:
* distanceToSensor.csv : File containing the list of relevant sensors and their distance to the scale.
* sensorVarDict.json : File containing the main variance in the signal. It is useful to identify the disturbances in the signal and generate the labeled dataset. Generated during pre-training, read during fine-tuning.

The dataset must be passed using the argument "--data_path".

### **Fine-tuning for TLE**

* Using the argument "--finetune", pass the path to the file *.pth of the pre-trained model, if not file is passed it will train a model from zero.
* Dataloader is the same from pre-training, however the flag ```isFineTuning = True``` is passed to call the fine-tune dataset.
