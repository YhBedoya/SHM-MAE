import cosdata as cs


def download_day(year, month, day):
    ss335 = cs.dataConnection('engineering', 'ss335')
    start = year + '-' + month + '-' + day + ' 00:00:00'
    end = year + '-' + month + '-' + day + ' 23:59:59'
    df = ss335.cosQuery(data_type='raw', sensor_type='acc', select='year,month,day,ts,sens_pos,x,y,z', start=start, end=end)
    if not(df.empty):
        df.to_csv(f'ss335-acc-{year}{month}{day}.csv')    


if __name__ == '__main__':
    for i in range(1, 31):
        download_day('2019', '4', str(i))
    for i in range(1, 32):
        download_day('2019', '5', str(i))
    for i in range(1, 31):
        download_day('2019', '6', str(i))
