import os
import json,time,sys
import datetime as dt
from datetime import datetime, timezone, timedelta
import s3fs
import ibmcloudsql
import collections
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd


credentials_cos_tech = {
  "apikey": "vtmeIYXwDGMVSnRvYcEi9nJH9hkMEcI2J8lvMKgfFOVa",
  "cos_hmac_keys": {
    "access_key_id": "9c276f7014894c00b3a9460856e84d76",
    "secret_access_key": "cca2abdc0ed8161dddeabe05345a6904a935aeaae8229e18"
  },
  "endpoints": "https://cos-service.bluemix.net/endpoints",
  "iam_apikey_description": "Auto generated apikey during resource-key operation for Instance - crn:v1:bluemix:public:cloud-object-storage:global:a/f596c0090891848de33f63a13544a19d:7dac646a-ed74-4a89-a325-51db059cf590::",
  "iam_apikey_name": "auto-generated-apikey-9c276f70-1489-4c00-b3a9-460856e84d76",
  "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Reader",
  "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/f596c0090891848de33f63a13544a19d::serviceid:ServiceId-783ab005-ec41-46bd-80ca-8fad67b4190e",
  "resource_instance_id": "crn:v1:bluemix:public:cloud-object-storage:global:a/f596c0090891848de33f63a13544a19d:7dac646a-ed74-4a89-a325-51db059cf590::",
  "endpoint_url" : "https://s3.eu-de.objectstorage.softlayer.net"
}

client_kwargs_tech = {"endpoint_url": credentials_cos_tech["endpoint_url"] }

high_freq_data_type=['acc','mic','mag']
low_freq_data_type=['tilt1','tilt2','tilt3','crack','fissure','stress','piezo',"vrms","vfft","lora_tilt"]

def daterange(start_date, end_date): #ciclo su giorni
    for n in range(int((end_date-start_date).days)):
        yield (start_date + timedelta(n))

class dataConnection:

    def __init__(self, verticalName, structName):
        '''
        '''
        self.vertical = verticalName # 'engineering'
        self.group = structName # 'dora_oulx'
        fs = s3fs.S3FileSystem(key=credentials_cos_tech["cos_hmac_keys"]["access_key_id"], secret=credentials_cos_tech["cos_hmac_keys"]["secret_access_key"], client_kwargs=client_kwargs_tech)
        l=fs.ls('installations/vertical=%s'%verticalName)
        struct_found=False

        for i in l:
          if 'config' in i:
            with fs.open(i, 'rb') as f:
              l=json.loads(f.read().decode('utf-8'))
            for installation in l['structures']:
              if installation['name']==structName:
                self.tenant = i.split('_')[-1].split('.')[0]
                self.bucket = installation['cos_bucket']
                self.cos_uri = 'cos://s3.eu-de.objectstorage.softlayer.net/'+installation['cos_bucket']+'/'
                self.data_types = installation['data_types']
                self.status = installation['status']
                self.cos_api_key_id = l['credentials'][2]['cos']['apikey']
                self.cos_service_instance_id = l['credentials'][2]['cos']['resource_instance_id']
                self.cos_access_key_id = l['credentials'][2]['cos']['cos_hmac_keys']['access_key_id']
                self.cos_secret_access_key = l['credentials'][2]['cos']['cos_hmac_keys']['secret_access_key']
                self.cos_endpoint = 'https://s3.eu-de.objectstorage.softlayer.net'
                self.sql_instance_crn = l['credentials'][4]['sql']['instance_crn']
                self.sql_platform_api_key = l['credentials'][4]['sql']['platform_api_key']
                self.sqlclient = ibmcloudsql.SQLQuery(self.sql_platform_api_key, self.sql_instance_crn)
                self.sqlclient.logon()
                self.client_kwargs = {"endpoint_url": self.cos_endpoint }
                self.fs = s3fs.S3FileSystem(key= self.cos_access_key_id, secret= self.cos_secret_access_key, client_kwargs= self.client_kwargs)
                struct_found=True

        if not struct_found:
          print(structName,'not found')
        else:
          print(structName,'connected')




    def cosQuery(self,data_type,sensor_type,select=None,start=None,end=None,sensors=None,source=None):
      if start is not None:
        ts_min=int(dt.datetime.strptime(start, '%Y-%m-%d %H:%M:%S').timestamp())*1000
      else:
        ts_min=int((dt.datetime.utcnow()-dt.timedelta(days=2*12*30)).timestamp()*1000)
      if end is not None:
        ts_max = int(dt.datetime.strptime(end, '%Y-%m-%d %H:%M:%S').timestamp())*1000
      else:
        ts_max = int(dt.datetime.utcnow().timestamp()*1000)

      if sensor_type=='acc':
        store = 'PARQUET'
      else:
        store = 'CSV'

      if select is not None:
        query = "SELECT "+ select +" FROM " + self.cos_uri + data_type + "/type="+sensor_type+"/ STORED AS PARQUET p1 WHERE"
      else:
        query = "SELECT * FROM " + self.cos_uri + data_type + "/type="+sensor_type+"/ STORED AS PARQUET p1 WHERE"
      query=query+" p1.ts>="+str(ts_min)+" AND p1.ts<="+str(ts_max)
      if source is not None:
        query=query+" AND p1.source='"+source+"'"
        if 'alarm' not in source:
          if ((start is not None) or (end is not None)):
            if start.split('-')[0]==end.split('-')[0]:
              query=query+" AND p1.year="+start.split('-')[0]
            if start.split('-')[1]==end.split('-')[1]:
              query=query+" AND p1.month="+start.split('-')[1]
              if ((start.split('-')[2].split(" ")[0]==end.split('-')[2].split(" ")[0]) & (sensor_type in high_freq_data_type)):
                query=query+" AND p1.day="+start.split('-')[2].split(" ")[0]
      if sensors is not None:
        query=query+" AND p1.sens_pos IN (" + ', '.join(str("\'"+x+"\'") for x in sensors)  + ")"

      query=query+" ORDER BY p1.ts ASC"
      query=query+" INTO " + self.cos_uri + 'query_results/' + " STORED AS "+ store

      response = self.sqlclient.run_sql(query)

      return response

    def cosList(self, path):
      bucket_uri = '{bucket}/{path}'.format(**{'bucket':self.bucket, 'path': path})
      files = self.fs.ls(bucket_uri)
      return files

    def cosUpload(self, df, path, verbose=True):
      ta=pa.Table.from_pandas(df.set_index('ts'))
      bucket_uri = '{bucket}/{path}'.format(**{'bucket':self.bucket, 'path': path})
      sink = self.fs.open(bucket_uri, 'wb')
      pw = pq.ParquetWriter(sink, schema=ta.schema)
      pw.write_table(ta)
      if verbose:
        print(path+' successfully stored')

    def cosDownload(self, path, format):
      bucket_uri = '{bucket}/{path}'.format(**{'bucket':self.bucket, 'path': path})
      data = self.fs.open(bucket_uri)
      if format == 'parquet':
        return pd.read_parquet(data)
      elif format == 'csv':
        return pd.read_csv(data)
      elif format == 'json':
        return json.load(data)
      else:
        print('format not supported')
        return None

    def cosDownloadMultiple(self,path,start_date,end_date):
      allfiles=self.cosList(path)
      days=[gg.strftime('%Y%m%d') for gg in daterange(start_date,end_date)]
      selected_days = [i for i in allfiles for d in days if d in i]
      df = pq.ParquetDataset(selected_days, filesystem=self.fs).read_pandas().to_pandas().sort_index()

      return df

    def cosDownloadAll(self,path):
      allfiles=self.cosList(path)
      print(allfiles)
      df = pq.ParquetDataset(allfiles, filesystem=self.fs).read_pandas().to_pandas().sort_index()

      return df
