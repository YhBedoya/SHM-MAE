import torch
import numpy as np
import math
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from scipy import signal
from torch.utils.data import Dataset
import random

import os
#os.chdir("/home/yhbedoya/Repositories/SHM-MAE")
import models_audio_mae_R
import glob
pd.options.mode.chained_assignment = None
import json

# define the utils

def prepare_model(chkpt_dir, arch='audioMae_vit_base'):
    # build model
    model = getattr(models_audio_mae_R, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(spectrogram, model, test):

    x = torch.tensor(spectrogram)

    x = x.unsqueeze(dim=0)

    y = model(x.float(), mask_ratio=0.8)

    return y

class SHMDataset(Dataset):

    def __init__(self, data_path, isPreTrain, isFineTuning, isEvaluation):
        if isPreTrain:
            self.start_time, self.end_time = "05/12/2021 00:00", "05/12/2021 23:59"
            self.datasetSize = 500000
        elif isFineTuning:
            self.start_time, self.end_time = "06/12/2021 00:00", "06/12/2021 11:59"
            self.datasetSize = 200000
        elif isEvaluation:
            self.start_time, self.end_time = "06/12/2021 12:00", "06/12/2021 17:59"
            self.datasetSize = 50000
        else:
            self.start_time, self.end_time = "06/12/2021 18:00", "06/12/2021 23:59"
            self.datasetSize = 500000
        self.path = data_path #'/home/yhbedoya/Repositories/SHM-MAE/traffic/20211205/'
        self.noisySensors = ["C12.1.4", "C17.1.2"]
        self.minDuration = 0.25
        self.data = self._readCSV()
        self.distanceToSensor = self._readDistanceToSensor()
        self.sensorVarDict = self._calculateThresholds(isPreTrain=isPreTrain)
        self.pesaDataDf = self._readLabels()
        self.labelsDf, self.groupsDf = self._labelAssignment()
        self.sampleRate = 100
        self.frameLength = 198
        self.stepLength = 58
        self.windowLength= 5990
        self.windowStep = 1500
        self.data, self.limits, self.totalWindows, min, max = self._partitioner()
        if isPreTrain:
            self.min = min
            self.max = max
        else:
            self.min = -24.496400092774163
            self.max = -3.3793421008937514

    def __len__(self):
        return self.totalWindows

    def __getitem__(self, index):
        start, end, label, timeSlice, sensor = self.limits[index]
        slice = self.data[start:end]
        frequencies, times, spectrogram = self._transformation(slice)
        spectrogram = torch.unsqueeze(torch.tensor(spectrogram, dtype=torch.float64), 0)
        NormSpect = self._normalizer(spectrogram).type(torch.float16)
        #print(f'type {type(NormSpect)}, inp shape: {slice.shape} out shape: {NormSpect.shape}')
        return torch.transpose(NormSpect, 1, 2), label

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
        df["zN"] = df["z"]-np.mean(df["z"])
        df["vars"] = df["zN"].rolling(window=100).var().fillna(0)
        df["vars"] = df["vars"].rolling(window=100).mean().fillna(0)
        print(f'finish reading process')
        return df

    def _readDistanceToSensor(self):
        distanceToSensor = {}
        with open('/home/yvelez/sacertis/distanceToSensor.csv') as f: #/home/yvelez/sacertis/distanceToSensor.csv
            for line in f.readlines():
                sensor, distance = line.replace("'", "").replace("\n","").split(",")
                distanceToSensor[sensor] = float(distance)
        return distanceToSensor

    def _readLabels(self):
        start_time = datetime.strptime(self.start_time, '%d/%m/%Y %H:%M')
        end_time = datetime.strptime(self.end_time, '%d/%m/%Y %H:%M')
        pesaDataDf = pd.read_csv("/home/yvelez/sacertis/dati_pese_dinamiche/dati 2021-12-04_2021-12-12 pesa km 104,450.csv", sep=";", index_col=0)
        pesaDataDf = pesaDataDf[["Id", "StartTimeStr", "ClassId", "GrossWeight", "Velocity", "VelocityUnit"]]
        pesaDataDf["Time"] = pd.to_datetime(pesaDataDf["StartTimeStr"])
        pesaDataDf["Time"] = pesaDataDf["Time"].dt.strftime('%Y-%d-%m %H:%M:00')
        pesaDataDf["Time"] = pd.to_datetime(pesaDataDf["Time"]) + pd.to_timedelta(-1,'H')
        pesaDataDf.sort_values(by="Id", inplace=True)
        pesaDataDf = pesaDataDf[(pesaDataDf["Time"]>=start_time) & (pesaDataDf["Time"]<=end_time)]
        pesaDataDf.reset_index(drop=True, inplace=True)
        
        return pesaDataDf

    def groupsGenerator(self, sensorData, minTime, maxTime, threshold):
        slice = sensorData[(sensorData["ts"]>= minTime) & (sensorData["ts"]<= maxTime)]
        
        slice["outlier"] = slice["vars"].apply(lambda x: x>=threshold)
        outliers = slice[slice["outlier"] == True].reset_index().to_dict("records")

        if len(outliers) == 0:
            return pd.DataFrame()

        last = minTime
        timeStart = outliers[0]["ts"]
        flag = True
        groups = []
        groupTimes = []
        groupIndexes = []
        groupVars = []
        label = np.nan
        groupId = 0
        for outlier in outliers:
            if ((outlier["ts"] - last).total_seconds() < 2) or flag:
                groupTimes.append(outlier["ts"])
                groupVars.append(outlier["vars"])
                flag = False
                timeEnd = outlier["ts"]
            else:
                start, end = min(groupTimes), max(groupTimes)
                groupSignal = sensorData[(sensorData["ts"]>= start) & (sensorData["ts"]<= end)]["zN"]
                signalPower = np.sqrt(np.mean(np.array(groupSignal)**2))**2 
                pointMaxVar = groupTimes[np.argmax(groupVars)]
                if ((end - start).total_seconds() > self.minDuration):
                    label = {"groupId": groupId,"start": start, "end": end, "signalPower": signalPower, 
                    "pointMaxVar": pointMaxVar}
                    groups.append(label)
                groupId += 1
                groupTimes = [outlier["ts"],]
                groupVars = [outlier["vars"],]
            last = outlier["ts"]

        start, end = min(groupTimes), max(groupTimes)
        groupSignal = sensorData[(sensorData["ts"]>= start) & (sensorData["ts"]<= end)]["zN"]
        signalPower = np.sqrt(np.mean(np.array(groupSignal)**2))**2 
        pointMaxVar = groupTimes[np.argmax(groupVars)]
        if ((end-start).total_seconds() > self.minDuration):
            label = {"groupId": groupId,"start": start, "end": end, "signalPower": signalPower, 
            "pointMaxVar": pointMaxVar}
            groups.append(label)

        if len(groups)>0:
            groupsDf = pd.DataFrame(groups).sort_values("signalPower", ascending=False)
        else:
            groupsDf = pd.DataFrame()

        return groupsDf

    def _labelAssignment(self,):
        sensorLabelsDfList = []
        groupsDfList = []

        sensorsList = self.data["sens_pos"].unique()
        for sensor in sensorsList:
            if (sensor in self.noisySensors) or (sensor not in self.distanceToSensor.keys()) or (sensor not in self.sensorVarDict.keys()):
                continue
            assignedLabels = {}
            assignedLabels2 = {}
            sensorLabelsDf = self.pesaDataDf.copy(deep=True)
            sensorLabelsDf["EstimatedTime"] = sensorLabelsDf["Time"] + pd.to_timedelta((float(self.distanceToSensor[sensor])/(sensorLabelsDf["Velocity"]/3.6))-20,'S')
            sensorLabelsDf["MaxTime"] = sensorLabelsDf["EstimatedTime"] + pd.to_timedelta(120,'S')
            minTime = sensorLabelsDf["EstimatedTime"].min()
            maxTime = sensorLabelsDf["MaxTime"].max()
            sensorLabelsDf.sort_values("GrossWeight", inplace=True, ascending=False)

            sensorData = self.data[self.data["sens_pos"]==sensor]
            threshold = self.sensorVarDict[sensor]["threshold"]

            groupsDf = self.groupsGenerator(sensorData, minTime, maxTime, threshold)
            print(f"Total groups found for sensor {sensor}: {groupsDf.shape[0]}")
            if groupsDf.empty:
                continue

            availableGroupsDf = groupsDf.copy(deep=True)
            for index, row in sensorLabelsDf.iterrows():
                if row["Id"] in assignedLabels:
                    continue
                
                if availableGroupsDf.empty:
                    break

                candidatesDf = availableGroupsDf[(row["EstimatedTime"] <= availableGroupsDf["pointMaxVar"]) & (availableGroupsDf["pointMaxVar"] <= row["MaxTime"])]
                if not candidatesDf.empty:
                    assignedLabels[row["Id"]] = candidatesDf.iloc[0].to_dict()
                    assignedLabels2[candidatesDf.iloc[0]["groupId"]] = row["Id"]
                    availableGroupsDf.drop(candidatesDf.index[0], inplace=True)
            
            sensorLabelsDf["sens_pos"] = sensor
            sensorLabelsDf["labels"] = sensorLabelsDf.apply(lambda row: assignedLabels[row["Id"]] if row["Id"] in assignedLabels else np.nan, axis=1)
            sensorLabelsDf.sort_values("Id", inplace=True)
            groupsDf["sens_pos"] = sensor
            groupsDf["labels"] = groupsDf.apply(lambda row: assignedLabels2[row["groupId"]] if row["groupId"] in assignedLabels2 else np.nan, axis=1)
            groupsDf.sort_values("groupId", inplace=True)
            groupsDf.dropna(inplace=True)

            sensorLabelsDfList.append(sensorLabelsDf)
            groupsDfList.append(groupsDf)

        labelsDf = pd.concat(sensorLabelsDfList)
        groupsDf =  pd.concat(groupsDfList)

        print(f"Total labels: {len(labelsDf)}")
        totnan = labelsDf["labels"].isna().sum()
        print(f"Total nan labels: {totnan}")
        print(f"Proportion of match labels: {1-(totnan/len(labelsDf))}")

        return labelsDf, groupsDf

    def _labelAssigner(self, timeSlice, sensor):
        start, end = timeSlice.min(), timeSlice.max()
        vehiclesInSliceDf = self.groupsDf[(self.groupsDf["pointMaxVar"]>=start) &
        (self.groupsDf["pointMaxVar"]<=end) &
        (self.groupsDf["sens_pos"]==sensor)]
        return vehiclesInSliceDf.shape[0]

    def _partitioner(self):
        sensors = self.data['sens_pos'].unique().tolist()
        print(f'start partitioner')
        partitions = {}
        cumulatedWindows = 0
        limits = dict()
        print(f'Generating windows')
        for sensor in tqdm(sensors):
            if (sensor in self.noisySensors) or (sensor not in self.distanceToSensor.keys()):
                continue
            sensorData = self.data[self.data['sens_pos']==sensor]
            totalFrames = sensorData.shape[0]
            totalWindows = math.ceil((totalFrames-self.windowLength)/self.windowStep)
            start = cumulatedWindows
            cumulatedWindows += totalWindows
            end = cumulatedWindows
            indexStart = sensorData.index[0]
            partitions[sensor]= (start, end, indexStart)

        timeData = torch.tensor(self.data["z"].values, dtype=torch.float64)
        timestamps = self.data["ts"]
        cummulator = -1

        mins = list()
        maxs = list()
        print(f'Defining useful windows limits')
        indexes = list(range(0, cumulatedWindows))
        random.shuffle(indexes)

        for index in tqdm(indexes):
            if cummulator >= self.datasetSize:
                break
            for sensor,v in partitions.items():
                if index in range(v[0], v[1]):
                    start = v[2]+(index-v[0])*self.windowStep
                    timeSlice = timestamps[start: start+self.windowLength]
                    label = self._labelAssigner(timeSlice, sensor)
                    filteredSlice = self.butter_bandpass_filter(timeData[start: start+self.windowLength], 0, 50, self.sampleRate)
                    signalPower = self.power(filteredSlice)

                    if (signalPower>1.25*10**-6) or (label>0):
                        cummulator += 1
                        limits[cummulator] = (start, start+self.windowLength, label, timeSlice, sensor)
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
        return np.sqrt(np.mean(np.array(slice)**2))**2

    def interquartileRule(self, data):
        # Calculate the first quartile (Q1)
        Q1 = np.percentile(data, 25)

        # Calculate the third quartile (Q3)
        Q3 = np.percentile(data, 75)

        # Calculate the interquartile range (IQR)
        IQR = Q3 - Q1

        # Define the upper and lower bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return lower_bound, upper_bound

    def _calculateThresholds(self, isPreTrain):
        if isPreTrain:
            print(f'Start creating thresholds')
            varDf = self.data[["sens_pos", "vars"]]
            sensorsList = self.data["sens_pos"].unique()
            sensorVarDict = {}
            for sensor in tqdm(sensorsList):
                if (sensor in self.noisySensors) or (sensor not in self.distanceToSensor.keys()):
                    continue
                sensorVarDf = varDf[varDf["sens_pos"]==sensor]
                lower_bound, upper_bound = self.interquartileRule(sensorVarDf["vars"])
                sensorVarDf = sensorVarDf[(sensorVarDf["vars"]>lower_bound) & (sensorVarDf["vars"]<upper_bound)]
                mean = sensorVarDf["vars"].mean()
                std = sensorVarDf["vars"].std()
                threshold = mean + 3.5 * std
                sensorVarDict[sensor] = {"mean": mean, "std": std, "threshold": threshold}
                #with open("/content/drive/MyDrive/Data Science and Engineering - PoliTo2/Thesis/models/MAE-SHM/output_dir_TE/sensorVarDict.json", "w") as f:
                #    # Write the dict to the file
                #    json.dump(sensorVarDict, f)
            print(f'Finish thresholds creation')
        else:
            print(f'Start reading thresholds')
            with open("/home/yvelez/sacertis/sensorVarDict.json", "r") as f:
                # Load the dict from the file
                sensorVarDict = json.load(f)

            print(f'Finish thresholds reading')

        return sensorVarDict

if __name__ == "__main__":

    version = "V2"
    path = f"/home/yvelez/SHM-MAE/output_dir_TE/fine_tuning{version}/"

    data_path = "/home/yvelez/sacertis/traffic/20211206/"
    testDataset = SHMDataset(data_path= data_path, 
    isPreTrain=False, isFineTuning=False, isEvaluation=False)

    checkpoints = []
    MSEs = []
    RMSEs = []
    MAEs = []

    for checkpoint in os.listdir(path):
        if checkpoint.split("-")[0] != "checkpoint":
            continue

        print(f"Starting ------ {checkpoint} ----")
        #Load the model
        chkpt_dir = path + checkpoint
        model_mae = prepare_model(chkpt_dir, 'audioMae_vit_base_R')
        print('Model loaded.')

        y = []
        y_pred = []

        for index in tqdm(range(len(testDataset))):
            dataPoint = testDataset[index]
            y.append(dataPoint[1])
            y_pred.append(run_one_image(dataPoint[0], model_mae, True)[0][0].detach().numpy())

        result = {
            "y_true": y,
            "y_pred": y_pred
        }

        MSE = ((np.array(y) - np.array(y_pred)) ** 2).mean()
        RMSE = np.sqrt(MSE)
        MAE = (np.abs(np.array(y) - np.array(y_pred))).mean()

        checkpoints.append(checkpoint)
        MSEs.append(MSE)
        RMSEs.append(RMSE)
        MAEs.append(MAE)

        resultsDf = pd.DataFrame(result)
        #saving results in npy files to read later
        resultsDf.to_csv(f'/home/yvelez/SHM-MAE/TrafficEstimation/TrafficEstimationResultsEvaluation{version}_{checkpoint}.csv')

    aggregator = {
    "checkpoints": checkpoints,
    "MSEs": MSEs,
    "RMSEs": RMSEs,
    "MAEs": MAEs
    }

    aggregationDf = pd.DataFrame(aggregator)
    #saving results in npy files to read later
    aggregationDf.to_csv(f'/home/yvelez/SHM-MAE/TrafficEstimation/TrafficEstimationResultsEvaluationAgg{version}.csv')
