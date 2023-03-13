from torch.utils.data import Dataset
import torch
import glob
import pandas as pd
pd.options.mode.chained_assignment = None
from datetime import datetime
import os
import math
from tqdm import tqdm
import numpy as np
import random

from scipy import signal
from scipy import stats
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

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
            self.start_time, self.end_time = "06/12/2021 17:59", "06/12/2021 23:59"
            self.datasetSize = 50000
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
        self.windowLength= 6000
        self.windowStep = 1500
        self.data, self.limits, self.totalWindows = self._partitioner()

    def __len__(self):
        return self.totalWindows

    def __getitem__(self, index):
        
        return self.limits[index]

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
                c_drop = set(df_tmp.columns) - set(["sens_pos", "x", "y", "z", "ts"])
                if len(c_drop) > 0:
                    df_tmp.drop(columns=list(c_drop), inplace=True)
                ldf.append(df_tmp)
        df = pd.concat(ldf).sort_values(by=['sens_pos', 'ts'])
        df.reset_index(inplace=True, drop=True)

        #df = df[df['sens_pos'].isin(self.sensors)]
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df['Time'] = df['ts'].dt.strftime('%Y-%m-%d %H:%M:00')
        df["xN"] = df["x"]-np.mean(df["x"])
        df["yN"] = df["y"]-np.mean(df["y"])
        df["zN"] = df["z"]-np.mean(df["z"])
        df["vars"] = df["zN"].rolling(window=100).var().fillna(0)
        df["vars"] = df["vars"].rolling(window=100).mean().fillna(0)
        print(f'finish reading process')
        return df

    def _readDistanceToSensor(self):
        distanceToSensor = {}
        with open("/home/yhbedoya/Repositories/SHM-MAE/LabelGeneration/distanceToSensor.csv") as f: #/home/yvelez/sacertis/distanceToSensor.csv  /home/yhbedoya/Repositories/SHM-MAE/LabelGeneration/distanceToSensor.csv
            for line in f.readlines():
                sensor, distance = line.replace("'", "").replace("\n","").split(",")
                distanceToSensor[sensor] = float(distance)
        return distanceToSensor

    def _readLabels(self):
        start_time = datetime.strptime(self.start_time, '%d/%m/%Y %H:%M')
        end_time = datetime.strptime(self.end_time, '%d/%m/%Y %H:%M')
        pesaDataDf = pd.read_csv("/home/yvelez/sacertis/dati_pese_dinamiche/dati 2021-12-04_2021-12-12 pesa km 104,450.csv", sep=";", index_col=0) #/home/yvelez/sacertis/dati_pese_dinamiche/dati 2021-12-04_2021-12-12 pesa km 104,450.csv  /home/yhbedoya/Repositories/SHM-MAE/dati_pese_dinamiche/dati 2021-12-04_2021-12-12 pesa km 104,450.csv
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
                    signalPower = self.power(timeData[start: start+self.windowLength])

                    if (signalPower>1.25*10**-6) or (label>0):
                        cummulator += 1
                        limits[cummulator] = (start, start+self.windowLength, label, (timestamps[start], timestamps[start+self.windowLength]), sensor)

                    break
        print(f'Total windows in dataset: {cummulator}')
        return timeData, limits, cummulator

    def _transformation(self, slice):
        sliceN = slice-torch.mean(slice)
        frequencies, times, spectrogram = signal.spectrogram(sliceN,self.sampleRate,nfft=self.frameLength,noverlap=(self.frameLength - self.stepLength), nperseg=self.frameLength,mode='psd')

        return frequencies, times, np.log10(spectrogram)
    
    def _normalizer(self, spectrogram):
        spectrogramNorm = (spectrogram - self.min) / (self.max - self.min)
        return spectrogramNorm

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
            with open("/home/yvelez/sacertis/sensorVarDict.json", "r") as f: #/home/yvelez/sacertis/sensorVarDict.json  /home/yhbedoya/Repositories/SHM-MAE/TrafficEstimation/sensorVarDict.json
                # Load the dict from the file
                sensorVarDict = json.load(f)

            print(f'Finish thresholds reading')

        return sensorVarDict

def groupAssigner(row, groupsDict):
    sensor = row["sensor"]
    for k, v in groupsDict.items():
        if sensor in v:
            return int(k)
        
def featureExtraction(window, sensNumber):
    statistics = dict()
    axis = ["xN", "yN", "zN"]
    for ax in axis:
        axSens = ax + sensNumber
        serie = window[ax]

        axStatistics={
            axSens+"mean": np.mean(serie),
            axSens+"std": np.std(serie),
            axSens+"min": np.min(serie),
            axSens+"max": np.max(serie),
            axSens+"med": np.median(serie),
            axSens+"kurt": stats.kurtosis(serie),
            axSens+"skew": stats.skew(serie),
            axSens+"rms": np.sqrt(np.mean(serie**2)),
            axSens+"sabs": np.sum(np.abs(serie)),
            axSens+"eom": serie[serie>np.mean(serie)].sum(),
            axSens+"ener": np.sqrt(np.mean(np.array(serie)**2))**2,
            axSens+"mad": np.median(np.absolute(serie - np.median(serie)))
        }

        statistics = {**statistics, **axStatistics}

    return statistics

def getDataset(training=False, evaluation=False, test=False):

    data_path = "/home/yvelez/sacertis/traffic/20211206/"  #  "/home/yvelez/sacertis/traffic/20211206/"  /home/yhbedoya/Repositories/SHM-MAE/traffic/20211206/
    dataGenerator = SHMDataset(data_path=data_path ,isPreTrain=False, isFineTuning=training, isEvaluation=evaluation)

    windows = {"indexStart":[],
        "indexEnd":[],
        "times": [],
        "label":[],
        "sensor":[]}

    for i in tqdm(range(len(dataGenerator))):
        start, end, label, timeSlice, sensor = dataGenerator[i]
        windows["indexStart"].append(start)
        windows["indexEnd"].append(end)
        windows["label"].append(label)
        windows["times"].append(timeSlice)
        windows["sensor"].append(sensor)

    df = dataGenerator._readCSV()
    df = df[["sens_pos", "ts", "xN", "yN", "zN"]]

    sensors = df["sens_pos"].unique()

    groupsDict = {}
    for sensor in list(sensors):
        section = sensor.split(".")[0]
        group = section[1:]

        if section[0] == "P":
            continue  
        if int(group) in groupsDict.keys():
            groupsDict[int(group)].append(sensor)
        else:
            groupsDict[int(group)] = [sensor]

    toDelete = list()
    for k, v in groupsDict.items():
        if len(v) != 6:
            toDelete.append(k)

    for k in toDelete:
        del groupsDict[k]
            
    windowsDf = pd.DataFrame(windows)
            
    windowsDf["group"] = windowsDf.apply(groupAssigner, groupsDict = groupsDict, axis=1)
    windowsDf.dropna(inplace=True)

    extractedFeaturesList = []
    group = 2
    groupDataDf = df[df["sens_pos"].isin(groupsDict[group])]
    groupWindowsDf = windowsDf[windowsDf["group"]==group]
    for index, row in tqdm(groupWindowsDf.iterrows()):
        statistics = dict()
        section = row["sensor"].split(".")[0]
        group = section[1:]
        
        if group =="8":
            continue
        window = groupDataDf[(groupDataDf["ts"] >= row["times"][0]) & (groupDataDf["ts"] < row["times"][1])]
        for sensor in groupsDict[int(group)]:
            sensorWindow =  window[window["sens_pos"]==row["sensor"]]
            sensorStatistics = featureExtraction(sensorWindow, sensor[-3:])
            statistics = {**statistics, **sensorStatistics}
        statistics["label"] = row["label"]

        extractedFeaturesList.append(statistics)

    dataDf = pd.DataFrame(extractedFeaturesList)
    return dataDf

def featureSelection(df, numFeatures):

    # Separate your features and target variable
    X = df.drop('label', axis=1)
    y = df['label']

    # Select the k best features using the F-test
    selector = SelectKBest(score_func=f_classif, k=numFeatures) # choose the number of features you want to keep
    X_new = selector.fit_transform(X, y)

    # Get the indices of the selected features
    mask = selector.get_support() # an array of booleans indicating which features are selected
    selected_features = list(X.columns[mask]) + ["label"] # a list of the selected feature names

    return selected_features

def dataSplit(trainDf, evalDf, testDf, selected_features):
    trainFDf = trainDf[selected_features]
    X = trainFDf.drop('label', axis=1)
    y = trainFDf['label']
    evalFDf = evalDf[selected_features]
    X_ev = evalFDf.drop('label', axis=1)
    y_ev = evalFDf['label']
    testFDf = testDf[selected_features]
    X_test = testFDf.drop('label', axis=1)
    y_test = testFDf['label']

    return X, y, X_ev, y_ev, X_test, y_test

def fitModels(models, X, y):

    for name, model in models.items():
        print(f"------- Starting training of {name} --------")
        model.fit(X, y)

    return models

def evaluateModels(trainedModels, X_test, y_test):

    results = {"model": [],
               "mse": [],
               "mae": [],
               "mape": [],
               "r2": []}

    for name, model in trainedModels.items():
        y_pred_test = model.predict(X_test)

        results["model"].append(name)
        results["mse"].append(mean_squared_error(y_test, y_pred_test))
        results["mae"].append(mean_absolute_error(y_test, y_pred_test))
        results["mape"].append((np.mean(np.abs(y_test - y_pred_test)) / np.mean(y_test)) * 100)
        results["r2"].append(r2_score(y_test, y_pred_test))

    return results

if __name__ == "__main__":

    trainDataset = getDataset(training=True, evaluation=False, test=False)
    evalDataset = getDataset(training=False, evaluation=True, test=False)
    testDataset = getDataset(training=False, evaluation=False, test=True)

    numFeaturesOptions = [5, 10, 15, 20, 25, 50, 75, 100, 150, 200, "all"]

    models = {"LR": LR(fit_intercept=True),
              "RF": RF(max_depth=200, n_estimators=30, verbose=2),
              "KNR": KNR(n_neighbors=7),
              "MLP": MLP(hidden_layer_sizes=(100, 100, 100), verbose=2),
              "SVR": SVR(kernel='rbf', C=10, verbose=2)}
    
    resultsDf = pd.DataFrame()
    
    for numFeatures in numFeaturesOptions:
        models = {"LR": LR(fit_intercept=True),
            "RF": RF(max_depth=200, n_estimators=30, verbose=2),
            "KNR": KNR(n_neighbors=7),
            "MLP": MLP(hidden_layer_sizes=(100, 100, 100), verbose=2),
            "SVR": SVR(kernel='rbf', C=10, verbose=2)}
        
        print(f"-------------- Start process with {numFeatures} features ---------------------")
        selected_features = featureSelection(trainDataset, numFeatures)
        X, y, X_ev, y_ev, X_test, y_test = dataSplit(trainDataset, evalDataset, testDataset, selected_features)

        trainedModels = fitModels(models, X, y)
        metricsDict = evaluateModels(trainedModels, X_test, y_test)

        metricsDf = pd.DataFrame(metricsDict)
        metricsDf["numFeatures"] = numFeatures

        resultsDf = pd.concat([resultsDf, metricsDf])

    resultsDf.to_csv("resultsTest.csv")



