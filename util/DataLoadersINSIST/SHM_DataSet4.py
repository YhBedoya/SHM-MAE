from torch.utils.data import Dataset
import torch
import glob
import pandas as pd
import datetime
import os
import math
from tqdm import tqdm
import numpy as np

from scipy import signal

from pathlib import Path

class SHMDataset(Dataset):

    def __init__(self, data_path):
        self.day_start = datetime.date(2019,5,10)
        self.num_days = 1
        self.path = data_path #Path("/home/yhbedoya/Repositories/SHM-MAE/INSIST_SS335/")
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
        return torch.transpose(NormSpect, 1, 2), 0

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

        for i in tqdm(range(len(df))):
            row = df["z"][i]
            data_splited = row.replace("\n", "").replace("[", "").replace("]", "").split(" ")
            #data_splited = df["z"][i].split(" ")
            ts = datetime.datetime.utcfromtimestamp(df["ts"][i]/1000)
            sens = df["sens_pos"][i]
            
            for idx, data in enumerate(data_splited):
                try:
                    if idx == 0:
                        z = int(data[1:])
                    elif idx == len(data_splited)-1:
                        z = int(data[:-1])
                    else:
                        z = int(data)
                except:
                    print(df["z"][i])
                    continue
                new_dict["ts"].append(ts + idx*datetime.timedelta(milliseconds=10))
                new_dict["z"].append(z * conv)
                new_dict["sens_pos"].append(sens)

        df_new = pd.DataFrame(new_dict)

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
