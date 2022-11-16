from torch.utils.data import Dataset
import torch
import glob
#import matplotlib.pyplot as plt
#import librosa.display
import numpy as np
import pandas as pd
from datetime import datetime
import os
import math
import time

import numpy as np
import matplotlib.pyplot as plt
import random

import librosa
import librosa.display
from scipy import signal
from tqdm import tqdm
import multiprocessing
import torchvision

class SHMDataset(Dataset):

    def __init__(self):
        self.start_time, self.end_time = "05/12/2021 23:59", "06/12/2021 00:00"
        self.path = "/home/yhbedoya/Repositories/SHM-MAE/traffic/"
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

                return frequencies, times, spectrogram

    def _readCSV(self):
        startTime = time.time()
        print(f'start reading')
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
        endTime = time.time()
        print(f"reading time: {endTime-startTime}")

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

    

def plotSpect(frequencies, times, spectrogram, index):
    print(spectrogram.shape)
    plt.figure(figsize=(10, 5))
    plt.title('spectrogram from PSD')
    plt.pcolormesh(times, frequencies, 10*np.log10(np.squeeze(spectrogram)), vmin=-150, vmax=-50)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(format="%+2.f", label='dB')
    plt.savefig(f'../spect/{index}.png')

def task(gen, i):
    frequencies, times, spectrogram = gen[i]
    means.append(np.mean(spectrogram))
    vars.append(np.var(spectrogram))



if __name__ == "__main__":
    timer = list()
    gen = SHMDataset()
    processes = []
    manager = multiprocessing.Manager()
    means = manager.list()
    vars = manager.list()

    #indexes = [random.randrange(0, 29820) for i in range(10)]
    indexes = range(0, len(gen))
    for i in tqdm(indexes):
        #print(f'Index: {i}')
        startMeasure = time.time()
        frequencies, times, spectrogram = gen[i]
        #means.append(np.mean(spectrogram))
        #vars.append(np.var(spectrogram))
        endMeasure = time.time()
        timer.append(endMeasure-startMeasure)

        #plotSpect(frequencies, times, spectrogram, i)
    #gnrMean = np.mean(np.array(means))
    #gnrStd = np.sqrt(np.mean(np.array(vars)))

    #startMeasure = time.time()
    #indexes = [random.randrange(0, 29820) for i in range(10)]
    #for i in tqdm(range(0, 10)):
    #    p = multiprocessing.Process(target = task, args=(gen, indexes[i]))
    #    p.start()
    #    processes.append(p)

    #for p in processes:
    #    p.join()
    #endMeasure = time.time()

    #print(f'Total time {endMeasure-startMeasure}')

    #gnrMean = np.mean(np.array(means))
    #gnrStd = np.sqrt(np.mean(np.array(vars)))
    #print(f'General mean {gnrMean} general std {gnrStd}')
