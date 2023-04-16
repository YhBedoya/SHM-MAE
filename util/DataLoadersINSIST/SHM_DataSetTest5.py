from torch.utils.data import Dataset
import torch
import glob
import matplotlib.pyplot as plt
#import librosa.display
import numpy as np
import pandas as pd
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
from pathlib import Path

import datetime

class SHMDataset(Dataset):

    def __init__(self):
        self.day_start = datetime.date(2019,5,24)
        self.num_days = 1
        self.path = "/home/yhbedoya/Repositories/SHM-MAE/INSIST_SS335/"
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
        start, end, power = self.limits[index]
        slice = self.data[start:end]
        frequencies, times, spectrogram = self._transformation(slice)
        spectrogram = torch.unsqueeze(torch.tensor(spectrogram, dtype=torch.float64), 0)
        NormSpect = self._normalizer(spectrogram).type(torch.float16)
        print(f'type {type(NormSpect)}, inp shape: {slice.shape} out shape: {NormSpect.shape}')
        return frequencies, times, spectrogram, power

    def _readCSV(self):
        print(f'reading CSV files')
        
        ldf = []
        for x in range(self.num_days):
            yy, mm, dd = (self.day_start + datetime.timedelta(days=x)).strftime('%Y,%m,%d').split(",")
            date = f"{int(yy)}{int(mm)}{int(dd)}"
            df = pd.read_csv(self.path + f"ss335-acc-{date}.csv")
            ldf.append(df.drop(['x','y', "year", "month", "day", "Unnamed: 0"], axis=1))
        df = pd.concat(ldf).sort_values(by=['sens_pos', 'ts'])
        df = df.reset_index(drop=True)

        new_dict = {
            "ts": [],
            "sens_pos": [],
            "z": [],
        }
        conv = (1*2.5)*2**-15

        print(f'Creating the dataframe')
        for i in tqdm(range(len(df))):
            row = df["z"][i]
            data_splited = row.replace("\n", "").replace("[", "").replace("]", "").split(" ")
            #data_splited = df["z"][i].split(" ")
            ts = datetime.datetime.utcfromtimestamp(df["ts"][i]/1000)
            sens = df["sens_pos"][i]
            
            for idx, data in enumerate(data_splited):
                if data == "":
                    continue
                z = int(data)  
                new_dict["ts"].append(ts + idx*datetime.timedelta(milliseconds=10))
                new_dict["z"].append(z * conv)
                new_dict["sens_pos"].append(sens)
            if i >100000:
                break

        df_new = pd.DataFrame(new_dict)
        print(f'Finish data reading')
        return df_new

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

        mins = list()
        maxs = list()
        print(f'Defining useful windows limits')
        noiseFreeSpaces = 0
        indexes = list(range(0, cumulatedWindows))
        random.shuffle(indexes)

        for index in tqdm(indexes):
            if cummulator >= 30000:
                break
            for k,v in partitions.items():
                if index in range(v[0], v[1]):
                    start = v[2]+(index-v[0])*self.windowStep
                    filteredSlice = timeData[start: start+self.windowLength]
                    signalPower = self.power(filteredSlice)

                    if signalPower>1.25*10**-6:
                        cummulator += 1
                        limits[cummulator] = (start, start+self.windowLength, signalPower)
                        slice = timeData[start:start+self.windowLength]
                        frequencies, times, spectrogram = self._transformation(torch.tensor(slice, dtype=torch.float64))
                        mins.append(np.min(np.array(spectrogram)))
                        maxs.append(np.max(np.array(spectrogram)))
                    break
        print(f'Total windows in dataset: {cummulator}')
        min = np.min(np.array(mins))
        max = np.max(np.array(maxs))   
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
    
    def power(self, slice):
        signalPower = np.sqrt(np.mean(np.array(slice)**2))**2
        return signalPower

    
def plotSpect(frequencies, times, spectrogram, index, power):
    plt.figure(figsize=(10, 5))
    plt.title(f'Spectrogram from PSD')
    times= np.linspace(1, 10, 80)
    plt.pcolormesh(times, frequencies, 10*(np.squeeze(spectrogram)), vmin=-150, vmax=-50)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(format="%+2.f", label='dB')
    #folder = "positives" if power > 1.25*10**-6 else "noise"
    #plt.savefig(f'/home/yhbedoya/Repositories/SHM-MAE/PowerINSIST/{folder}/{index}.png')
    #plt.close()

def task(gen, i):
    frequencies, times, spectrogram, power = gen[i]
    #means.append(np.mean(spectrogram))
    #vars.append(np.var(spectrogram))
    plotSpect(frequencies, times, spectrogram, i, power)


if __name__ == "__main__":
    timer = list()
    gen = SHMDataset()
    processes = []
    manager = multiprocessing.Manager()
    means = manager.list()
    vars = manager.list()

    print(f"Total instances: {len(gen)}")

    indexes = [random.randrange(0, len(gen)) for i in range(10)]
    #indexes = range(0,len(gen))
    #maxs = []
    for i in tqdm(indexes):

        frequencies, times, spectrogram, power = gen[i]
        if power > 1.25*10**-6:
            plotSpect(frequencies, times, spectrogram, i, power)

    #startMeasure = time.time()
    #indexes = [random.randrange(0, len(gen)) for i in range(10000)]
    #indexes = range(56700, 56900)
    #batchSize = 100
    #batches = math.floor(len(indexes)/batchSize)
    #for batchNumber in tqdm(range(0, batches)):
    #    start= batchSize*batchNumber
    #    indexBatch = range(start,start+batchSize)
    #    for i in indexBatch:
    #        p = multiprocessing.Process(target = task, args=(gen, indexes[i]))
    #        p.start()
    #        processes.append(p)

    #    for p in processes:
    #        p.join()
    #endMeasure = time.time()

    #print(f'Total time {endMeasure-startMeasure}')

    #gnrMean = np.mean(np.array(means))
    #gnrStd = np.sqrt(np.mean(np.array(vars)))
    #print(f'General mean {gnrMean} general std {gnrStd}')


