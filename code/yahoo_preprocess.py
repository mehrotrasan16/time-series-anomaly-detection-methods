a1_csv = glob.glob(f'../data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/**/*.csv', recursive=True)
print(f'Loaded the paths of {len(a1_csv)} files from disk. Begin processing at: {start_time}')
for index,file in enumerate(a1_csv):
if index%10 == 0:
    print(f'Processing index: {index} of {len(a1_csv)}')
if index > 1:
     break
fname = file.split("/")[5].replace('\\','').split(".")[0]
df = pd.read_csv(file)
df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
df_indexed = df.set_index('timestamp')

print("Data:")
print(df_indexed.head())

scaler2 = StandardScaler()
stddf = df_indexed.copy()
for col in stddf.columns:
    if col not in ["timestamp","is_anomaly"]:
        stddf[col] = scaler2.fit_transform(np.reshape(df_indexed[col].values,(len(df_indexed[col].values),1)))

print('Mean: %f, StandardDeviation: %f' % (scaler2.mean_, sqrt(scaler2.var_)))
print("Standardized values:")
print(stddf.head()) 

scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = scaler.fit(np.reshape(masked[col].values,(len(masked[col]),1)))        
scaleddf = stddf.copy()
for col in scaleddf.columns:
    if col not in ["timestamp","is_anomaly"]:
        scaleddf[col] = scaler.fit_transform(np.reshape(df_indexed[col].values,(len(df_indexed[col].values),1)))
print("Normalized values:")
print(scaleddf.head())

print("\n\n\n")