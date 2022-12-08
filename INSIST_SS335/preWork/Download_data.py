import cosdata as cs
import pandas as pd
import numpy as np
from datetime import date
from datetime import timedelta
from tqdm import tqdm

# Explodes the lists in DF rows for multiple columns
def unnesting(df, explode, axis):
    if axis==1:
        idx = df.index.repeat(df[explode[0]].str.len())
        df1 = pd.concat([pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
        df1.index = idx
        return df1.join(df.drop(explode, 1), how='left')
    else :
        df1 = pd.concat([pd.DataFrame(df[x].tolist(), index=df.index).add_prefix(x) for x in explode], axis=1)
        return df1.join(df.drop(explode, 1), how='left')

def daterange(start_date, end_date): #ciclo su giorni
    for n in range(int((end_date-start_date).days)):
        yield (start_date + timedelta(n))
'''
def unnesting(df, explode, axis):
    if axis==1:
        df = df.reset_index()
        idx = df.index.repeat(df[explode[0]].str.len())
        df1 = pd.concat([pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
        df1.index = idx
        return pd.merge(df1, df['ts'], how='inner', left_index=True, right_index=True, sort = False)
    else :
        df1 = pd.concat([pd.DataFrame(df[x].tolist(), index=df.index).add_prefix(x) for x in explode], axis=1)
        return df1.join(df.drop(explode, 1), how='left')
'''
def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )
'''
Dora = cs.dataConnection('engineering','dora_oulx')
response = Dora.cosDownloadAll('derived/type=tilt2/algo=mov_avg')
response.to_csv('Dora-tilt2-with-mean.csv')

Dora = cs.dataConnection('engineering','dora_oulx')
result1=Dora.cosQuery('raw','tilt2', source ='periodic')
result1.to_csv('Dora-tilt2.csv')
'''
ss335 = cs.dataConnection('engineering','ss335')
#result1=ss335.cosQuery('raw','tilt2', source = 'periodic')
#result1.to_csv('ss335-tilt2.csv')
start_date=date(2019,5,21)
end_date=date(2019,5,25)
for gg in tqdm([date for date in daterange(start_date,end_date)]):
    y,m,d=(gg.strftime('%Y,%m,%d')).split(",")
    start =y+'-'+ m +'-'+d+' 00:00:00'
    end = y+'-'+ m +'-'+d+' 23:59:59'
    result1=ss335.cosQuery(data_type='raw', sensor_type='acc', select='x,y,z,ts,sens_pos', start=start, end=end)
    #result1=ss335.cosQuery('raw','acc','x,y,z,ts,T,sens_pos',start,end, sensors = ['S6.1.3'], source='stream_20190522')
    #result1 = result1.drop(['month', 'day', 'year', 'source','group', 'H', 'ts0', 'ts2', 'sens_zone','sn'], 1)
    start =y+'-'+ m +'-'+d
    if not(result1.empty):
        for sens in result1['sens_pos'].unique():
            temp = result1[result1['sens_pos']==sens]
            df = unnesting(temp,['x','y','z'],axis=1)
            #df.to_parquet('ss335-acc'+start+sens.replace(".","")+'.parquet')
            df.to_csv('ss335-acc'+start+sens.replace(".","")+'.csv')
#Dora = cs.dataConnection('engineering','dora_oulx')
#result1=Dora.cosQuery('derived','tilt2', alg = 'mov'source ='periodic')
# result1.to_csv('Dora-tilt2.csv')
# valbona = cs.dataConnection('engineering','valbona_molinazzo')
# result1=valbona.cosQuery('raw','tilt2',source ='periodic')
# result1.to_csv('Valbona-tilt2.csv')
# perilleux = cs.dataConnection('engineering','perilleux')
# result1=perilleux.cosQuery('raw','tilt2',source ='periodic')
# result1.to_csv('Perilleux-tilt2.csv')
