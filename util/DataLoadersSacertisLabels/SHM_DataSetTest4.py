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
        self.start_time, self.end_time = "05/12/2021 23:00", "06/12/2021 00:00"
        self.path = '/home/yhbedoya/Repositories/SHM-MAE/traffic/20211205/'
        self.data = self._readCSV()
        self.labelsDf = self._readLabels()
        self.data = self._labelMatch()
        self.sampleRate = 100
        self.frameLength = 256
        self.stepLength = 64
        self.windowLength= 6000
        self.windowStep = 1500
        self.data, self.limits, self.totalWindows, self.min, self.max = self._partitioner()

    def __len__(self):
        return self.totalWindows

    def __getitem__(self, index):
        start, end, std, label = self.limits[index]
        slice = self.data[start:end]
        frequencies, times, spectrogram = self._transformation(slice)
        spectrogram = torch.unsqueeze(torch.tensor(spectrogram, dtype=torch.float64), 0)
        NormSpect = self._normalizer(spectrogram).type(torch.float16)
        #print(f'type {type(NormSpect)}, inp shape: {slice.shape} out shape: {NormSpect.shape}')
        return frequencies, times, spectrogram, std, label

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
        df['Time'] = df['ts'].dt.strftime('%Y-%m-%d %H:%M:00')

        return df

    def _readLabels(self):
        pesaDataDf = pd.read_csv("/home/yhbedoya/Repositories/SHM-MAE/dati_pese_dinamiche/dati 2021-12-04_2021-12-12 pesa km 104,450.csv", sep=";", index_col=0)
        pesaDataDf = pesaDataDf[["Id", "StartTimeStr"]]
        pesaDataDf["Time"] = pd.to_datetime(pesaDataDf["StartTimeStr"])
        pesaDataDf["Time"] = pesaDataDf["Time"].dt.strftime('%Y-%d-%m %H:%M:00')
        pesaDataDf["Time"] = pd.to_datetime(pesaDataDf["Time"]) + pd.to_timedelta(-1,'H') 
        aggDf = pesaDataDf.groupby(["Time"])["Id"].count()
        labelsDf = aggDf.reset_index()
        labelsDf["Time"] = pd.to_datetime(labelsDf["Time"]).dt.strftime('%Y-%m-%d %H:%M:00')
        labelsDf.rename(columns={"Id": "Vehicles"}, inplace=True)
        
        return labelsDf

    def _labelMatch(self):
        dataDf = self.data
        labelsDf = self.labelsDf

        dataLabelsDf = dataDf.set_index("Time").join(labelsDf.set_index("Time"), how="left").reset_index()
        dataLabelsDf["Vehicles"].fillna(0, inplace=True)

        return dataLabelsDf

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
        vehiclesData = self.data["Vehicles"]
        cummulator = -1

        mins = list()
        maxs = list()
        print(f'Defining useful windows limits')
        indexes = list(range(0, cumulatedWindows))
        random.shuffle(indexes)

        for index in tqdm(indexes):
            if cummulator >= 500000:
                break
            for k,v in partitions.items():
                if index in range(v[0], v[1]):
                    start = v[2]+(index-v[0])*self.windowStep
                    filteredSlice = self.butter_bandpass_filter(timeData[start: start+self.windowLength], 0, 50, self.sampleRate)
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
    
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        return signal.butter(order, [1, 49], fs=fs, btype='band')

    def butter_bandpass_filter(self, slice, lowcut, highcut, fs, order=5):
        sliceN = slice-np.mean(np.array(slice))
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, sliceN)
        return y

    def power(self, slice):
        signalPower = np.sqrt(np.mean(np.array(slice)**2))**2
        return signalPower

    
def plotSpect(frequencies, times, spectrogram, index, std, label):
    plt.figure(figsize=(10, 5))
    plt.title(f'spectrogram from PSD: {round(std, 4)} Vehicles: {label}')
    plt.pcolormesh(times, frequencies, 10*(np.squeeze(spectrogram)), vmin=-150, vmax=-50)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(format="%+2.f", label='dB')
    folder = "positives" if std > 0.0075 else "noise"
    plt.savefig(f'/home/yhbedoya/Repositories/SHM-MAE/spectOneM/{folder}/{index}.png')
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

    indexes = [random.randrange(0, len(gen)) for i in range(5000)]
    #indexes = range(0,len(gen))
    for i in tqdm(indexes):
        frequencies, times, spectrogram, std, label = gen[i]


        plotSpect(frequencies, times, spectrogram, i, std, label)

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


