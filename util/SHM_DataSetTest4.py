from torch.utils.data import Dataset
import torch
import glob
import matplotlib.pyplot as plt
#import librosa.display
import numpy as np
import pandas as pd
from datetime import datetime
import os
import math
import time
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
        self.start_time, self.end_time = "05/12/2021 14:00", "06/12/2021 14:30"
        self.path = '/home/yhbedoya/Repositories/SHM-MAE/subTraffic/'
        self.data = self._readCSV()
        self.sampleRate = 100
        self.frameLength = 198
        self.stepLength = 10
        self.windowLength= 990
        self.windowStep = 100
        self.data, self.limits, self.totalWindows, self.min, self.max = self._partitioner()

    def __len__(self):
        return self.totalWindows

    def __getitem__(self, index):
        start, end, std = self.limits[index]
        slice = self.data[start:end]
        frequencies, times, spectrogram = self._transformation(slice)
        spectrogram = torch.unsqueeze(torch.tensor(spectrogram, dtype=torch.float64), 0)
        NormSpect = self._normalizer(spectrogram).type(torch.float16)
        #print(f'type {type(NormSpect)}, inp shape: {slice.shape} out shape: {NormSpect.shape}')
        return frequencies, times, spectrogram, std

    def _readCSV(self):
        print(f'reading CSV files')
        start = datetime.strptime(self.start_time, '%d/%m/%Y %H:%M')
        end = datetime.strptime(self.end_time, '%d/%m/%Y %H:%M')

        ldf = list()
        for p in tqdm(glob.glob(self.path + "*.csv")):
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

    def _partitioner(self):
        sensors = self.data['sens_pos'].unique().tolist()
        print(f'start partitioner')
        partitions = {}
        cumulatedWindows = 0
        limits = dict()
        print(f'Generating windows')
        for sensor in tqdm(sensors):
            sensorData = self.data[self.data['sens_pos']==sensor]
            totalFrames = sensorData.shape[0]
            totalWindows = math.ceil((totalFrames-self.windowLength)/self.windowStep)
            start = cumulatedWindows
            cumulatedWindows += totalWindows
            end = cumulatedWindows
            indexStart = sensorData.index[0]
            partitions[sensor]= (start, end, indexStart)

        timeData = torch.tensor(self.data["z"].values, dtype=torch.float64)
        cummulator = -1
        posCummulator = 0
        negCummulator = 0


        mins = list()
        maxs = list()
        print(f'Defining useful windows limits')
        noiseFreeSpaces = 1
        for index in tqdm(range(0, cumulatedWindows)):
            for k,v in partitions.items():
                if index in range(v[0], v[1]):
                    start = v[2]+(index-v[0])*self.windowStep
                    filteredSlice = self.butter_bandpass_filter(timeData[start: start+self.windowLength], 0, 50, self.sampleRate)
                    amp = np.max(filteredSlice)-np.min(filteredSlice)
                    if amp > 0.0075:
                        posCummulator +=1 
                        cummulator += 1
                        limits[cummulator] = (start, start+self.windowLength, amp)
                        slice = timeData[start:start+self.windowLength]
                        frequencies, times, spectrogram = self._transformation(torch.tensor(slice, dtype=torch.float64))
                        mins.append(np.min(np.array(spectrogram)))
                        maxs.append(np.max(np.array(spectrogram)))
                        noiseFreeSpaces += 1
                        
                    elif noiseFreeSpaces>0:
                        negCummulator +=1
                        cummulator += 1
                        limits[cummulator] = (start, start+self.windowLength, amp)
                        slice = timeData[start:start+self.windowLength]
                        frequencies, times, spectrogram = self._transformation(torch.tensor(slice, dtype=torch.float64))
                        mins.append(np.min(np.array(spectrogram)))
                        maxs.append(np.max(np.array(spectrogram)))
                        noiseFreeSpaces -= 1
                    break
        print(f'Total windows in dataset: {cummulator}')
        min = np.min(np.array(mins))
        max = np.max(np.array(maxs))
        print(f'Total positive instances: {posCummulator}')
        print(f'Total noisy instances: {negCummulator}')
        print(f'Proportion of useful instances {(posCummulator+negCummulator)/cumulatedWindows}')       
        print(f'General min: {min}')
        print(f'General max: {max}')
        return timeData, limits, cummulator, min, max

    def _transformation(self, slice):
        sliceN = slice-torch.mean(slice)
        frequencies, times, spectrogram = signal.spectrogram(sliceN,self.sampleRate,nfft=self.frameLength,noverlap=(self.frameLength - self.stepLength), nperseg=self.frameLength,mode='psd')

        return frequencies, times, np.log10(spectrogram)
    
    def _normalizer(self, spectrogram):
        spectrogramNorm = (spectrogram - self.min) / (self.max - self.min)
        return spectrogramNorm
    
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        return signal.butter(order, [1, 49], fs=fs, btype='band')

    def butter_bandpass_filter(self, slice, lowcut, highcut, fs, order=5):
        sliceN = slice-np.mean(np.array(slice))
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, sliceN)
        return y

    
def plotSpect(frequencies, times, spectrogram, index, std):
    plt.figure(figsize=(10, 5))
    plt.title(f'spectrogram from PSD: {std}')
    plt.pcolormesh(times, frequencies, 10*(np.squeeze(spectrogram)), vmin=-150, vmax=-50)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(format="%+2.f", label='dB')
    folder = "positives" if std > 0.0075 else "noise"
    plt.savefig(f'../spect/{folder}/{index}.png')
    plt.close()

def task(gen, i):
    frequencies, times, spectrogram, std = gen[i]
    #means.append(np.mean(spectrogram))
    #vars.append(np.var(spectrogram))
    plotSpect(frequencies, times, spectrogram, i, std)


if __name__ == "__main__":
    timer = list()
    gen = SHMDataset()
    processes = []
    manager = multiprocessing.Manager()
    means = manager.list()
    vars = manager.list()

    #indexes = [random.randrange(0, len(gen)) for i in range(50000)]
    #indexes = range(0,len(gen))
    #maxs = []
    #for i in tqdm(indexes):
        #print(f'Index: {i}')
        #startMeasure = time.time()
    #    frequencies, times, spectrogram, std = gen[i]
    #    maxs.append(torch.max(spectrogram))
        #means.append(np.mean(spectrogram))
        #vars.append(np.var(spectrogram))
        #endMeasure = time.time()
        #timer.append(endMeasure-startMeasure)
    #plt.hist(maxs, bins=50, alpha=0.5, label='Maximums dist')
    #plt.ylim([0,10])
    #plt.xlim([0,2])
    #plt.show()

    #    plotSpect(frequencies, times, spectrogram, i, std)
    #gnrMean = np.mean(np.array(means))
    #gnrStd = np.sqrt(np.mean(np.array(vars)))

    #startMeasure = time.time()
    indexes = [random.randrange(0, len(gen)) for i in range(10000)]
    #indexes = range(56700, 56900)
    batchSize = 100
    batches = math.floor(len(indexes)/batchSize)
    for batchNumber in tqdm(range(0, batches)):
        start= batchSize*batchNumber
        indexBatch = range(start,start+batchSize)
        for i in indexBatch:
            p = multiprocessing.Process(target = task, args=(gen, indexes[i]))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    #endMeasure = time.time()

    #print(f'Total time {endMeasure-startMeasure}')

    #gnrMean = np.mean(np.array(means))
    #gnrStd = np.sqrt(np.mean(np.array(vars)))
    #print(f'General mean {gnrMean} general std {gnrStd}')


