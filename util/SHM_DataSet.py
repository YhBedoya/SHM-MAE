from torch.utils.data import Dataset
import torch
import glob
import torchaudio
import numpy as np
import pandas as pd
from datetime import datetime
import os
import math
import time
import torchvision

from scipy import signal

class SHMDataset(Dataset):

    def __init__(self, data_path):
        self.start_time, self.end_time = "05/12/2021 23:30", "06/12/2021 00:00"
        self.path = data_path #'/home/yhbedoya/Repositories/SHM-MAE/traffic/'
        self.data = self._readCSV()
        self.sampleRate = 100
        self.frameLength = 128
        self.stepLength = 8
        self.windowLength= 1000
        self.windowStep = 100
        self.sensors = self._getSensors()
        self.partitions, self.totalWindows = self._partitioner()
        self.Normalizer = torchvision.transforms.Normalize(mean=[3.900204491107504e-09], std=[5.970413343698724e-08])

    def __len__(self):
        return self.totalWindows

    def __getitem__(self, index):
        for k,v in self.partitions.items():
            if index in range(v[0], v[1]):
                sensorData = self.data[self.data['sens_pos']==k]
                start = sensorData.index[0]+(index-v[0])*self.windowStep
                slice = self.data.iloc[start:start+self.windowLength]["z"]
                frequencies, times, spectrogram = self._transformation(slice)
                spectrogram = torch.unsqueeze(torch.tensor(spectrogram), 0)
                NormSpect = self.Normalizer(spectrogram)
                print(f'index: {index}, type {NormSpect}')
                return NormSpect, None

    def _readCSV(self):
        start = datetime.strptime(self.start_time, '%d/%m/%Y %H:%M')
        end = datetime.strptime(self.end_time, '%d/%m/%Y %H:%M')

        ldf = list()
        for p in glob.glob(self.path + "*.csv"):
            name = os.path.split(p)[-1]
            nstr = datetime.strptime(name, 'traffic_%Y%m%dH%H%M%S.csv')
            if start <= nstr < end:
                df_tmp = pd.read_csv(p)
                c_drop = set(df_tmp.columns) - set(["sens_pos", "z", "ts"])
                if len(c_drop) > 0:
                    df_tmp.drop(columns=list(c_drop), inplace=True)
                ldf.append(df_tmp)
        df = pd.concat(ldf).sort_values(by=['sens_pos', 'ts'])
        df.reset_index(inplace=True, drop=True)

        #df = df[df['sens_pos'].isin(self.sensors)]
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')

        return df

    def _getSensors(self):
        sensors = self.data['sens_pos'].unique().tolist()

        return sensors

    def _partitioner(self):
        startTime = time.time()
        print(f'start partitioner')
        partitions = {}
        cumulatedWindows = 0
        for sensor in self.sensors:
            measurements  = self.data[self.data['sens_pos']==sensor]
            totalFrames = measurements.shape[0]
            totalWindows = math.ceil((totalFrames-self.windowLength)/self.windowStep)
            start = cumulatedWindows
            cumulatedWindows += totalWindows
            end = cumulatedWindows
            partitions[sensor]= (start, end)
            
        print(f'Total number of data points {cumulatedWindows}')
        endTime = time.time()
        print(f"partitioner time: {endTime-startTime}")
        return partitions, cumulatedWindows

    def _transformation(self, slice):
        
        sliceN = slice-np.mean(slice)
        frequencies, times, spectrogram = signal.spectrogram(sliceN,self.sampleRate,nfft=self.frameLength,noverlap=(self.frameLength - self.stepLength), nperseg=self.frameLength,mode='psd')

        return frequencies, times, spectrogram


