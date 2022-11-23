from torch.utils.data import Dataset
import torch
import glob
import pandas as pd
from datetime import datetime
import os
import math
import torchvision
from tqdm import tqdm

from scipy import signal

class SHMDataset(Dataset):

    def __init__(self, data_path):
        self.start_time, self.end_time = "05/12/2021 00:00", "06/12/2021 00:00"
        self.path = data_path #'/home/yhbedoya/Repositories/SHM-MAE/traffic/'
        self.data = self._readCSV()
        self.sampleRate = 100
        self.frameLength = 198
        self.stepLength = 10
        self.windowLength= 990
        self.windowStep = 100
        self.data, self.limits, self.totalWindows, self.gnrMean, self.gnrStd = self._partitioner()
        self.Normalizer = torchvision.transforms.Normalize(mean=[self.gnrMean], std=[self.gnrStd])

    def __len__(self):
        return self.totalWindows

    def __getitem__(self, index):
        start, end = self.limits[index]
        slice = self.data[start:end]
        frequencies, times, spectrogram = self._transformation(slice)
        spectrogram = torch.unsqueeze(torch.tensor(spectrogram, dtype=torch.float64), 0)
        NormSpect = self.Normalizer(spectrogram).type(torch.float16)
        #print(f'type {type(NormSpect)}, inp shape: {slice.shape} out shape: {NormSpect.shape}')
        return torch.transpose(NormSpect, 1, 2), 0

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
        for sensor in sensors:
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

        means = list()
        vars = list()
        print(f'Defining windows limits')
        noiseFreeSpaces = 1
        for index in tqdm(range(0, cumulatedWindows)):
            for k,v in partitions.items():
                if index in range(v[0], v[1]):
                #print(f'index: {index}')
                    start = v[2]+(index-v[0])*self.windowStep
                    std = torch.std(timeData[start: start+self.windowLength])
                    if std > 0.001:
                        cummulator += 1
                        limits[cummulator] = (start, start+self.windowLength)
                        slice = timeData[start:start+self.windowLength]
                        frequencies, times, spectrogram = self._transformation(torch.tensor(slice, dtype=torch.float64))
                        means.append(torch.mean(torch.tensor(spectrogram, dtype=torch.float64)))
                        vars.append(torch.var(torch.tensor(spectrogram, dtype=torch.float64)))
                        noiseFreeSpaces += 1
                        
                    elif noiseFreeSpaces > 0:
                        cummulator += 1
                        limits[cummulator] = (start, start+self.windowLength)
                        slice = timeData[start:start+self.windowLength]
                        frequencies, times, spectrogram = self._transformation(torch.tensor(slice, dtype=torch.float64))
                        means.append(torch.mean(torch.tensor(spectrogram, dtype=torch.float64)))
                        vars.append(torch.var(torch.tensor(spectrogram, dtype=torch.float64)))
                        noiseFreeSpaces -= 1
                    break
        print(f'Total windows in dataset: {cummulator}')
        gnrMean = torch.mean(torch.tensor(means, dtype=torch.float64))
        gnrStd = torch.sqrt(torch.mean(torch.tensor(vars, dtype=torch.float64)))
        print(f'General dataset mean: {gnrMean}')
        print(f'General dataset std: {gnrStd}')
        return timeData, limits, cummulator, gnrMean, gnrStd

    def _transformation(self, slice):
        
        sliceN = slice-torch.mean(slice)
        frequencies, times, spectrogram = signal.spectrogram(sliceN,self.sampleRate,nfft=self.frameLength,noverlap=(self.frameLength - self.stepLength), nperseg=self.frameLength,mode='psd')

        return frequencies, times, spectrogram


