95/7: runfile('D:/Temp/time-series/nasa_data_preprocessing.py', wdir='D:/Temp/time-series')
95/8: runfile('D:/Temp/time-series/nasa_data_preprocessing.py', wdir='D:/Temp/time-series')
95/9: runfile('D:/Temp/time-series/nasa_data_preprocessing.py', wdir='D:/Temp/time-series')
95/10: runfile('D:/Temp/time-series/nasa_data_preprocessing.py', wdir='D:/Temp/time-series')
95/11: runfile('D:/Temp/time-series/nasa_data_preprocessing.py', wdir='D:/Temp/time-series')
95/12: runfile('D:/Temp/time-series/nasa_data_preprocessing.py', wdir='D:/Temp/time-series')
95/13: runfile('D:/Temp/time-series/nasa_data_preprocessing.py', wdir='D:/Temp/time-series')
95/14: runfile('D:/Temp/time-series/nasa_data_preprocessing.py', wdir='D:/Temp/time-series')
95/15: runcell(0, 'C:/Users/Sanke/AppData/Local/Temp/kite_tutorial.py')
96/1: runfile('D:/Temp/time-series/nasa_data_preprocessing.py', wdir='D:/Temp/time-series')
96/2: runfile('D:/Temp/time-series/nasa_data_preprocessing.py', wdir='D:/Temp/time-series')
96/3: runfile('D:/Temp/time-series/nasa_data_preprocessing.py', wdir='D:/Temp/time-series')
96/4: runfile('D:/Temp/time-series/nasa_data_preprocessing.py', wdir='D:/Temp/time-series')
96/5: runfile('D:/Temp/time-series/nasa_data_preprocessing.py', wdir='D:/Temp/time-series')
96/6: runfile('D:/Temp/time-series/untitled0.py', wdir='D:/Temp/time-series')
96/7: runfile('D:/Temp/time-series/untitled0.py', wdir='D:/Temp/time-series')
96/8: runfile('D:/Temp/time-series/untitled0.py', wdir='D:/Temp/time-series')
96/9: runfile('D:/Temp/time-series/untitled0.py', wdir='D:/Temp/time-series')
96/10: runfile('D:/Temp/time-series/untitled0.py', wdir='D:/Temp/time-series')
96/11: runfile('D:/Temp/time-series/untitled0.py', wdir='D:/Temp/time-series')
96/12: runfile('D:/Temp/time-series/untitled0.py', wdir='D:/Temp/time-series')
96/13: runfile('D:/Temp/time-series/untitled0.py', wdir='D:/Temp/time-series')
96/14: runfile('D:/Temp/time-series/untitled0.py', wdir='D:/Temp/time-series')
96/15: runfile('D:/Temp/time-series/untitled0.py', wdir='D:/Temp/time-series')
96/16: runfile('D:/Temp/time-series/untitled0.py', wdir='D:/Temp/time-series')
96/17: runfile('D:/Temp/time-series/yahoo_a2_preproc.py', wdir='D:/Temp/time-series')
96/18: runfile('D:/Temp/time-series/yahoo_a2_preproc.py', wdir='D:/Temp/time-series')
96/19: runfile('D:/Temp/time-series/yahoo_a2_preproc.py', wdir='D:/Temp/time-series')
96/20: runfile('D:/Temp/time-series/yahoo_a2_preproc.py', wdir='D:/Temp/time-series')
96/21: runfile('D:/Temp/time-series/yahoo_a2_preproc.py', wdir='D:/Temp/time-series')
96/22: runfile('D:/Temp/time-series/yahoo_a2_preproc.py', wdir='D:/Temp/time-series')
96/23: runfile('D:/Temp/time-series/nasa_data_preprocessing.py', wdir='D:/Temp/time-series')
96/24: runfile('D:/Temp/time-series/nasa_data_preprocessing.py', wdir='D:/Temp/time-series')
96/25: runfile('D:/Temp/time-series/yahoo_a2_preproc.py', wdir='D:/Temp/time-series')
96/26: runfile('D:/Temp/time-series/yahoo_a2_preproc.py', wdir='D:/Temp/time-series')
96/27: result_add.plot()
96/28: runfile('D:/Temp/time-series/yahoo_a2_preproc.py', wdir='D:/Temp/time-series')
96/29: runfile('D:/Temp/time-series/yahoo_a2_preproc.py', wdir='D:/Temp/time-series')
96/30: threshold
96/31: residualdf = result_add.resid
96/32: type(residualdf)
96/33: residualdf
96/34: residualdf[residualdf > threshold]
96/35: runfile('D:/Temp/time-series/yahoo_a2_preproc.py', wdir='D:/Temp/time-series')
96/36: runfile('D:/Temp/time-series/yahoo_a2_preproc.py', wdir='D:/Temp/time-series')
96/37: runfile('D:/Temp/time-series/yahoo_a2_preproc.py', wdir='D:/Temp/time-series')
97/1:
import pandas as pd
from matplotlib.pyplot import pyplot as plt
import os
from statsmodels.tsa.seasonal import seasonal_decompose
97/2:
import pandas as pd
from matplotlib import pyplot as plt
import os
from statsmodels.tsa.seasonal import seasonal_decompose
97/3:
import pandas as pd
from matplotlib import pyplot as plt
import os
from statsmodels.tsa.seasonal import seasonal_decompose
import glob
import datetime
97/4:
start_time = datetime.now()
all_json = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/**/*.csv', recursive=True)
end_time = datetime.now()
print(f'Loaded the paths of {len(all_json)} files from disk. Took {end_time-start_time}')
97/5:
import pandas as pd
from matplotlib import pyplot as plt
import os
from statsmodels.tsa.seasonal import seasonal_decompose
import glob
from datetime import datetime
97/6:
start_time = datetime.now()
all_json = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/**/*.csv', recursive=True)
end_time = datetime.now()
print(f'Loaded the paths of {len(all_json)} files from disk. Took {end_time-start_time}')
97/7: all_csv[0]
97/8:
start_time = datetime.now()
all_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/**/*.csv', recursive=True)
end_time = datetime.now()
print(f'Loaded the paths of {len(all_csv)} files from disk. Took {end_time-start_time}')
97/9: all_csv[0]
97/10:
start_time = datetime.now()
all_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/**/*.csv', recursive=True)
end_time = datetime.now()
print(f'Loaded the paths of {len(all_csv)} files from disk. Took {end_time-start_time}')
97/11: all_csv[0]
97/12:
pd.read_csv(all_csv[0])
df
97/13:
df= pd.read_csv(all_csv[0])
df
97/14: df.describe()
97/15: df.info()
97/16: df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
97/17: df
97/18: df['timestamp'] = pd.to_datetime(df['timestamp'],unit='ns')
97/19: df
97/20: df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s',format='%f')
97/21: df
97/22:
df_indexed = df.set_index('timestamp')

print(df_indexed.info())
97/23:
# Additive Decomposition
result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
97/24:
# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
#result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()
97/25:
# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
#result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().subtitle('Additive Decompose', fontsize=22)
plt.show()
97/26:
# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
#result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()
97/27:
threshold=500
residualdf = result_add.resid
outliers = residualdf[residualdf > threshold]
print("File: data\\yahoo\\dataset\\ydata-labeled-time-series-anomalies-v1_0\\A2Benchmark\\synthetic_1.csv")
print("threshold: 500")
print("Outliers:")
print(outliers)
97/28:
threshold=500
residualdf = result_add.resid
outliers = residualdf[residualdf > threshold]
print(f"File: {all_csv[0]}")
print("threshold: 500")
print("Outliers:")
print(outliers)
97/29:
for file in all_csv:
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("A2Benchmark_" + file.split("\\")[1] +"add_STL")
    plt.show()
97/30:
for file in all_csv:
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("A2Benchmark_" + file.split(".")[0].split("\\")[1] +"add_STL")
    plt.show()
97/31:
for file in all_csv:
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("A2Benchmark_" + file.split("\\")[1].split(".")[0] +"add_STL")
    plt.show()
97/32:
start_time = datetime.now() 
for index,file in enumerate(all_csv):
    if index%10 ==0:
        print(f'Processing index: {index} of {len(all_csv)}')
    fname = file.split("\\")[1].split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("A2Benchmark_" + fname +"add_STL")
    print("\n\n\nA2Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    with open(fname, 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("threshold: 500")
        file.write("Outliers")
        file.writelines(outliers)
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print("STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:" + end_time-start_time)
97/33:
start_time = datetime.now() 
for index,file in enumerate(all_csv):
    if index%10 ==0:
        print(f'Processing index: {index} of {len(all_csv)}')
    fname = file.split("\\")[1].split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("A2Benchmark_" + fname +"add_STL")
    print("\n\n\nA2Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    with open(fname, 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("threshold: 500")
        file.write("Outliers")
        file.write(outliers)
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print("STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:" + end_time-start_time)
97/34:
start_time = datetime.now() 
for index,file in enumerate(all_csv):
    if index%10 ==0:
        print(f'Processing index: {index} of {len(all_csv)}')
    fname = file.split("\\")[1].split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("A2Benchmark_" + fname +"add_STL")
    print("\n\n\nA2Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    with open("./output/"+ fname, 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("threshold: 500")
        file.write("Outliers")
        file.write(outliers.to_csv())
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print("STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:" + end_time-start_time)
97/35:
start_time = datetime.now() 
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 15:
         break
    fname = file.split("\\")[1].split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("A2Benchmark_" + fname +"add_STL")
    print("\n\n\nA2Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    with open("./output/"+ fname, 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print("STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:" + end_time-start_time)
97/36:
start_time = datetime.now() 
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 15:
         break
    fname = file.split("\\")[1].split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("A2Benchmark_" + fname +"add_STL")
    print("\n\n\nA2Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    with open("./output/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print("STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:" + end_time-start_time)
97/37:
start_time = datetime.now() 
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 15:
         break
    fname = file.split("\\")[1].split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("A2Benchmark_" + fname +"add_STL")
    print("\n\n\nA2Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    with open("./STLoutput/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print("STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:" + end_time-start_time)
97/38: print(f"STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
96/38: runfile('D:/Temp/time-series/yahoo_a2_preproc.py', wdir='D:/Temp/time-series')
96/39: runfile('D:/Temp/time-series/yahoo_a2_preproc.py', wdir='D:/Temp/time-series')
96/40: runfile('D:/Temp/time-series/yahoo_a2_preproc.py', wdir='D:/Temp/time-series')
96/41: runfile('D:/Temp/time-series/yahoo_a2_preproc.py', wdir='D:/Temp/time-series')
98/1: runfile('D:/Temp/time-series/yahoo_a2_preproc.py', wdir='D:/Temp/time-series')
98/2: runfile('D:/Temp/time-series/yahoo_a2_preproc.py', wdir='D:/Temp/time-series')
97/39:
start_time = datetime.now() 
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 15:
         break
    fname = file.split("\\")[1].split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("./STLoutput/A2Benchmark_" + fname +"add_STL")
    print("\n\n\nA2Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    with open("./STLoutput/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print(f"STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
97/40:
start_time = datetime.now() 
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 50:
         break
    fname = file.split("\\")[1].split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("./STLoutput/A2Benchmark_" + fname +"add_STL")
    print("\n\n\nA2Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    with open("./STLoutput/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print(f"STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
97/41:
a1_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/**/*.csv', recursive=True)
start_time = datetime.now() 
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 50:
         break
    fname = file.split("\\")[1].split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("./STLoutput/A2Benchmark_" + fname +"add_STL")
    print("\n\n\nA2Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    with open("./STLoutput/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print(f"STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
97/42:
a1_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/**/*.csv', recursive=True)
start_time = datetime.now() 
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 50:
         break
    fname = file.split("\\")[1].split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("./STLoutput/A1Benchmark_" + fname +"add_STL")
    print("\n\n\nA1Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    with open("./STLoutput/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print(f"STL Additive decomposition of Yahoo S5 A1 Benchmark processing complete. Time taken:{end_time-start_time}")
97/43: all_csv.split("/")
97/44: all_csv[0].split("/")
97/45: all_csv[0].split("/")[5].replace('\\','').split()
97/46: all_csv[0].split("/")[5].replace('\\','').split(".")
97/47: all_csv[0].split("/")[5].replace('\\','').split(".")[0]
97/48:
start_time = datetime.now() 
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 50:
         break
    fname = file.split("/")[5].replace('\\','').split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("./STLoutput/A2Benchmark_" + fname +"add_STL")
    print("\n\n\nA2Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    with open("./STLoutput/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print(f"STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
97/49:
all_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/**/*.csv', recursive=True)
start_time = datetime.now() 
print(f'There are {len(all_csv)} to process, beginning at {start_time}')
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 50:
         break
    fname = file.split("\\")[1].split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("./STLoutput/A1Benchmark_" + fname +"add_STL")
    print("\n\n\nA1Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    with open("./STLoutput/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print(f"STL Additive decomposition of Yahoo S5 A1 Benchmark processing complete. Time taken:{end_time-start_time}")
97/50:
all_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A3Benchmark/**/*.csv', recursive=True)
start_time = datetime.now() 
print(f'There are {len(all_csv)} to process, beginning at {start_time}')
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 50:
         break
    fname = file.split("\\")[1].split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("./STLoutput/A1Benchmark_" + fname +"add_STL")
    print("\n\n\nA1Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    with open("./STLoutput/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print(f"STL Additive decomposition of Yahoo S5 A1 Benchmark processing complete. Time taken:{end_time-start_time}")
97/51:
all_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A4Benchmark/**/*.csv', recursive=True)
start_time = datetime.now() 
print(f'There are {len(all_csv)} to process, beginning at {start_time}')
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 50:
         break
    fname = file.split("\\")[1].split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("./STLoutput/A1Benchmark_" + fname +"add_STL")
    print("\n\n\nA1Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    with open("./STLoutput/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print(f"STL Additive decomposition of Yahoo S5 A1 Benchmark processing complete. Time taken:{end_time-start_time}")
97/52:
all_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/**/*.csv', recursive=True)
start_time = datetime.now() 
print(f'There are {len(all_csv)} to process, beginning at {start_time}')
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 50:
         break
    fname = file.split("\\")[1].split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("./STLoutput/A1Benchmark_" + fname +"add_STL")
    print("\n\n\nA1Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    with open("./STLoutput/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print(f"STL Additive decomposition of Yahoo S5 A1 Benchmark processing complete. Time taken:{end_time-start_time}")
97/53:
a1_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/**/*.csv', recursive=True)
start_time = datetime.now() 
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 50:
         break
    fname = file.split("/")[5].replace('\\','').split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("./STLoutput/A1Benchmark_" + fname +"_add_STL")
    print("\n\n\nA1Benchmark_" + fname +"_add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    with open("./STLoutput/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print(f"STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
99/1:
import pandas as pd
from matplotlib import pyplot as plt
import os
from statsmodels.tsa.seasonal import seasonal_decompose
import glob
from datetime import datetime
99/2:
start_time = datetime.now()
all_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/**/*.csv', recursive=True)
end_time = datetime.now()
print(f'Loaded the paths of {len(all_csv)} files from disk. Took {end_time-start_time}')
99/3: all_csv[0]
99/4: all_csv[0].split("/")[5].replace('\\','').split(".")[0]
99/5:
df= pd.read_csv(all_csv[0])
df
99/6: df.describe()
99/7: df.info()
99/8: df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
99/9: df
99/10:
df_indexed = df.set_index('timestamp')

print(df_indexed.info())
99/11:
# Additive Decomposition
result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
99/12:
# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
#result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()
99/13:
threshold=500
residualdf = result_add.resid
outliers = residualdf[residualdf > threshold]
print(f"File: {all_csv[0]}")
print("threshold: 500")
print("Outliers:")
print(outliers)
99/14:
start_time = datetime.now() 
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 50:
         break
    fname = file.split("/")[5].replace('\\','').split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("./STLoutput/A2Benchmark_" + fname +"add_STL")
    print("\n\n\nA2Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    with open("./STLoutput/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print(f"STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
99/15: print(f"STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
99/16:
a1_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/**/*.csv', recursive=True)
start_time = datetime.now() 
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 50:
         break
    fname = file.split("/")[5].replace('\\','').split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("./STLoutput/A1Benchmark_" + fname +"_add_STL")
    print("\n\n\nA1Benchmark_" + fname +"_add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    with open("./STLoutput/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print(f"STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
99/17:
a3_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A3Benchmark/A3Benchmark-TS*.csv', recursive=True)
start_time = datetime.now() 
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 50:
         break
    fname = file.split("/")[5].replace('\\','').split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("./STLoutput/A3Benchmark_" + fname +"_add_STL")
    print("\n\n\nA3Benchmark_" + fname +"_add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    with open("./STLoutput/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print(f"STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
99/18:
a3_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A3Benchmark/A4Benchmark-TS*.csv', recursive=True)
start_time = datetime.now() 
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 50:
         break
    fname = file.split("/")[5].replace('\\','').split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("./STLoutput/A3Benchmark_" + fname +"_add_STL")
    print("\n\n\nA3Benchmark_" + fname +"_add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    with open("./STLoutput/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print(f"STL Additive decomposition of Yahoo S5 A4 Benchmark processing complete. Time taken:{end_time-start_time}")
101/1: runfile('D:/Temp/Python_GIS_Map/county_combiner.py', wdir='D:/Temp/Python_GIS_Map')
101/2: runfile('D:/Temp/Python_GIS_Map/county_combiner.py', wdir='D:/Temp/Python_GIS_Map')
101/3: all_counties_colo[0]
101/4: runfile('D:/Temp/Python_GIS_Map/county_combiner.py', wdir='D:/Temp/Python_GIS_Map')
105/1: from ..ref_code.TSFeatures import TSFeatures
106/1: from ..ref_code.TSFeatures import TSFeatures
106/2:  print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
102/1: runfile('C:/Users/Sanke/untitled0.py', wdir='C:/Users/Sanke')
102/2: runfile('C:/Users/Sanke/untitled0.py', wdir='C:/Users/Sanke')
106/3: print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
106/4: import code.ref_code.TSFeatures
106/5: import TSFeatures
106/6: import TSFeatures
106/7:
import rpy2
print(rpy2.__version__)
106/8: import rpy2.robjects as robjects
106/9:
import rpy2
print(rpy2.__version__)
106/10: import rpy2.robjects as robjects
106/11: import rpy2.robjects as robjects
106/12:
import os
os.environ['R_HOME'] = "C:\Users\Sanke\anaconda3\Lib\R\bin\x64\R.dll"
106/13:
import os
os.environ['R_HOME']="C:/Users/Sanke/anaconda3/Lib/R/bin/x64\R.dll"
106/14:
import rpy2
print(rpy2.__version__)
106/15: import rpy2.robjects as robjects
106/16:
import os
os.environ['R_HOME']="C:/Users/Sanke/anaconda3/Lib/R/bin/x64/R.dll"
106/17:
import rpy2
print(rpy2.__version__)
106/18: import rpy2.robjects as robjects
106/19:
import os
os.environ['R_HOME']="C:/Users/Sanke/anaconda3/Lib/R/"
106/20:
import rpy2
print(rpy2.__version__)
106/21: import rpy2.robjects as robjects
109/1:
import os
os.environ['R_HOME']="C:/Users/Sanke/anaconda3/Lib/R/"
109/2:
import rpy2
print(rpy2.__version__)
109/3: import rpy2.robjects as robjects
111/1: import os
111/2: import rpy2
110/1: import rpy2.robjects as robjects
111/3: from rpy2 import robjects
111/4: from rpy2 import robjects
111/5: import rpy2
111/6: from rpy2 import robjects
112/1: import os
112/2: import rpy2
112/3: from rpy2 import robjectts
112/4: from rpy2 import robjects
112/5: runfile('D:/Temp/time-series/code/ref_code/TSFeatures.py', wdir='D:/Temp/time-series/code/ref_code')
112/6: runfile('D:/Temp/time-series/code/ref_code/TSFeatures.py', wdir='D:/Temp/time-series/code/ref_code')
112/7: runfile('D:/Temp/time-series/code/ref_code/TSFeatures.py', wdir='D:/Temp/time-series/code/ref_code')
114/1:
def extract_features(self, timeseries):
        oddstream=importr('oddstream')

        #r_timeseries = pandas2ri.py2ri(timeseries)
        with localconverter(ro.default_converter + pandas2ri.converter):
            for col in timeseries.columns.values:
                timeseries[col]=timeseries[col].astype(str) 
            #r_timeseries = ro.conversion.py2rpy(timeseries)
            features=oddstream.extract_tsfeatures(timeseries)
            #features= ro.conversion.rpy2py(features)
            return features
        return []
114/2:
def extract_features(timeseries):
        oddstream=importr('oddstream')

        #r_timeseries = pandas2ri.py2ri(timeseries)
        with localconverter(ro.default_converter + pandas2ri.converter):
            for col in timeseries.columns.values:
                timeseries[col]=timeseries[col].astype(str) 
            #r_timeseries = ro.conversion.py2rpy(timeseries)
            features=oddstream.extract_tsfeatures(timeseries)
            #features= ro.conversion.rpy2py(features)
            return features
        return []
114/3:
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
114/4:
def extract_features(timeseries):
        oddstream=importr('oddstream')

        #r_timeseries = pandas2ri.py2ri(timeseries)
        with localconverter(ro.default_converter + pandas2ri.converter):
            for col in timeseries.columns.values:
                timeseries[col]=timeseries[col].astype(str) 
            #r_timeseries = ro.conversion.py2rpy(timeseries)
            features=oddstream.extract_tsfeatures(timeseries)
            #features= ro.conversion.rpy2py(features)
            return features
        return []
114/5:
import pandas as pd
from matplotlib import pyplot as plt
import os
from statsmodels.tsa.seasonal import seasonal_decompose
import glob
from datetime import datetime
114/6:
start_time = datetime.now()
all_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/**/*.csv', recursive=True)
end_time = datetime.now()
print(f'Loaded the paths of {len(all_csv)} files from disk. Took {end_time-start_time}')
114/7: all_csv[0]
114/8: all_csv[0].split("/")[5].replace('\\','').split(".")[0]
114/9:
df= pd.read_csv(all_csv[0])
df
114/10: df.describe()
114/11: df.info()
114/12: df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
114/13: df
114/14:
df_indexed = df.set_index('timestamp')

print(df_indexed.info())
114/15:
# Additive Decomposition
result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
114/16:
# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
#result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()
114/17:
threshold=500
residualdf = result_add.resid
outliers = residualdf[residualdf > threshold]
print(f"File: {all_csv[0]}")
print("threshold: 500")
print("Outliers:")
print(outliers)
114/18: df_indexed
114/19: extract_features(df_indexed)
114/20:
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter

def extract_features(self, timeseries):
    oddstream=importr('oddstream')

    #r_timeseries = pandas2ri.py2ri(timeseries)
    with localconverter(ro.default_converter + pandas2ri.converter):
        for col in timeseries.columns.values:
            timeseries[col]=timeseries[col].astype(str) 
        #r_timeseries = ro.conversion.py2rpy(timeseries)
        features=oddstream.extract_tsfeatures(timeseries)
        #features= ro.conversion.rpy2py(features)
        return features
    return []
114/21:
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter

def extract_features(self, timeseries):
    try:
        oddstream=importr('oddstream')
    except:
        robjects.r(f'install.packages("{package_name}")')
        pkg = importr(package_name)

    #r_timeseries = pandas2ri.py2ri(timeseries)
    with localconverter(ro.default_converter + pandas2ri.converter):
        for col in timeseries.columns.values:
            timeseries[col]=timeseries[col].astype(str) 
        #r_timeseries = ro.conversion.py2rpy(timeseries)
        features=oddstream.extract_tsfeatures(timeseries)
        #features= ro.conversion.rpy2py(features)
        return features
    return []
114/22: df_indexed
114/23: extract_features(df_indexed)
114/24:
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter

def extract_features(timeseries):
    try:
        oddstream=importr('oddstream')
    except:
        robjects.r(f'install.packages("{package_name}")')
        pkg = importr(package_name)

    #r_timeseries = pandas2ri.py2ri(timeseries)
    with localconverter(ro.default_converter + pandas2ri.converter):
        for col in timeseries.columns.values:
            timeseries[col]=timeseries[col].astype(str) 
        #r_timeseries = ro.conversion.py2rpy(timeseries)
        features=oddstream.extract_tsfeatures(timeseries)
        #features= ro.conversion.rpy2py(features)
        return features
    return []
114/25: df_indexed
114/26: extract_features(df_indexed)
114/27:
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter

def extract_features(timeseries):
    try:
        oddstream=importr('oddstream')
    except:
        robjects.r(f'install.packages("oddstream")')
        pkg = importr('oddstream')

    #r_timeseries = pandas2ri.py2ri(timeseries)
    with localconverter(ro.default_converter + pandas2ri.converter):
        for col in timeseries.columns.values:
            timeseries[col]=timeseries[col].astype(str) 
        #r_timeseries = ro.conversion.py2rpy(timeseries)
        features=oddstream.extract_tsfeatures(timeseries)
        #features= ro.conversion.rpy2py(features)
        return features
    return []
114/28: df_indexed
114/29: extract_features(df_indexed)
114/30:
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter

def extract_features(timeseries):
    try:
        oddstream=importr('oddstream')
    except:
        robjects.r(f'install.packages("oddstream")')
        oddstream = importr('oddstream')

    #r_timeseries = pandas2ri.py2ri(timeseries)
    with localconverter(ro.default_converter + pandas2ri.converter):
        for col in timeseries.columns.values:
            timeseries[col]=timeseries[col].astype(str) 
        #r_timeseries = ro.conversion.py2rpy(timeseries)
        features=oddstream.extract_tsfeatures(timeseries)
        #features= ro.conversion.rpy2py(features)
        return features
    return []
114/31: df_indexed
114/32: extract_features(df_indexed)
114/33: features = extract_features(df_indexed)
114/34: type(features)
114/35: pandas2ri.ri2py(features)
114/36: type(pandas2ri.ri2py(features))
114/37:
features = extract_features(df_indexed)
features
114/38:
features = extract_features(df_indexed)
print(features)
114/39: features[0]
112/8: runfile('D:/Temp/time-series/code/nasa_data_preprocessing.py', wdir='D:/Temp/time-series/code')
112/9: runfile('D:/Temp/time-series/code/nasa_data_preprocessing.py', wdir='D:/Temp/time-series/code')
112/10: runfile('D:/Temp/time-series/code/nasa_data_preprocessing.py', wdir='D:/Temp/time-series/code')
114/40: result_add
112/11: runfile('D:/Temp/time-series/STL Decomposition for Time Series Anomalous Record Detection.py', wdir='D:/Temp/time-series')
112/12: runfile('D:/Temp/time-series/STL Decomposition for Time Series Anomalous Record Detection.py', wdir='D:/Temp/time-series')
114/41: residualdf
112/13: result_add
112/14: result_add.trend
112/15: outliers
112/16: fname
112/17: result_add.resid
112/18: result_add.resid[result_add.resid > threshold]
112/19: df_index
112/20: df_indexed
112/21: outliers
112/22: df_indexed.join(outliers,on='timestamp')
112/23: df_indexed.join(outliers,on='timestamp',type='right')
112/24: df_indexed.join(outliers,on='timestamp',how='right')
112/25: result_check = df_indexed.join(outliers,on='timestamp',how='right')
112/26: result_check.loc[result_check.is_anomaly == 1]
112/27: result_check.loc[result_check.is_anomaly == 0]
112/28: result_check.loc[result_check['is_anomaly'] == 0]
112/29: result_check.loc[result_check['is_anomaly'] > 0]
112/30: result_check.loc[result_check['is_anomaly'] == '1']
112/31: result_check.loc[result_check['is_anomaly'] == '0']
112/32:
df_indexed.loc[df_indexed['is_anomaly'] == 
'1']
112/33:
count(df_indexed.loc[df_indexed['is_anomaly'] == 
'1'])
112/34: truepositives = result_check.loc[result_check['is_anomaly'] == '1']
112/35: df_indexed.join(outliers,on='timestamp',how='inner')
112/36: result_check = df_indexed.join(outliers,on='timestamp',how='inner')
112/37: truepositives.count
112/38: truepositives.count()
112/39: falsepositives = anomalies.loc[anomalies['is_anomaly'] == '0']
112/40:
outliers = result_add.resid[result_add.resid > threshold]
not_outliers = result_add.resid[result_add.resid < threshold]

anomalies = df_indexed.join(outliers,on='timestamp',how='inner')

not_anomalies = df_indexed.join(not_outliers,on='timestamp',how='inner')
112/41:
truepositives = anomalies.loc[anomalies['is_anomaly'] == '1']

falsepositives = anomalies.loc[anomalies['is_anomaly'] == '0']

truenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == '0']

falsenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == '1']
112/42: truepositives.count()
112/43: falsepositives.count()
112/44: falsepositives.count() + 1
114/42:
outliers = result_add.resid[result_add.resid > threshold]
not_outliers = result_add.resid[result_add.resid < threshold]
114/43:
anomalies = df_indexed.join(outliers,on='timestamp',how='inner')
not_anomalies = df_indexed.join(not_outliers,on='timestamp',how='inner')
114/44:
p = df_indexed.loc[df_indexed['is_anomaly'] == '1']

n = df_indexed.loc[df_indexed['is_anomaly'] == '0']
114/45:
truepositives = anomalies.loc[anomalies['is_anomaly'] == '1']

falsepositives = anomalies.loc[anomalies['is_anomaly'] == '0']
114/46:
truenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == '0']

falsenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == '1']
114/47:
#Traditional FPR and TPR formmulae
#tpr = truepositives.count()/(truepositives.count() + falsenegatives.count())
#fpr = falsepositives.count()/(falsepositives.count() + truenegatives.count())

#IDEAL Paper based TRP/FPR rates
fpr = falsepositives.count()/n.count()
tpr = truepositives.count()/p.count()
fnr = 1-tpr
tnr = 1-fpr
114/48:
precision = truepositives.count()/(tpr + fpr)
recall = tpr/(tpr + fnr)
114/49:
start_time = datetime.now()
all_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/**/*.csv', recursive=True)
end_time = datetime.now()
print(f'Loaded the paths of {len(all_csv)} files from disk. Took {end_time-start_time}')
114/50: all_csv[0]
114/51: all_csv[0].split("/")[5].replace('\\','').split(".")[0]
114/52:
df= pd.read_csv(all_csv[0])
df
114/53: df.describe()
114/54: df.info()
114/55: df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
114/56: df
114/57:
df_indexed = df.set_index('timestamp')

print(df_indexed.info())
114/58:
# Additive Decomposition
result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
114/59:
# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
#result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()
114/60:
threshold=500
residualdf = result_add.resid
outliers = residualdf[residualdf > threshold]
print(f"File: {all_csv[0]}")
print("threshold: 500")
print("Outliers:")
print(outliers)
114/61:
outliers = residualdf[residualdf > threshold]
not_outliers = residualdf[residualdf < threshold]
outliers.info()
not_outliers.info()
114/62:
outliers = residualdf[residualdf > threshold]
not_outliers = residualdf[residualdf < threshold]
print(outliers,not_outliers)
114/63:
anomalies = df_indexed.join(outliers,on='timestamp',how='inner')
not_anomalies = df_indexed.join(not_outliers,on='timestamp',how='inner')
114/64:
anomalies = df_indexed.join(outliers,on='timestamp',how='inner')
not_anomalies = df_indexed.join(not_outliers,on='timestamp',how='inner')
print(anomalies,not_anomalies)
114/65:
p = df_indexed.loc[df_indexed['is_anomaly'] == '1']

n = df_indexed.loc[df_indexed['is_anomaly'] == '0']
114/66:
truepositives = anomalies.loc[anomalies['is_anomaly'] == '1']

falsepositives = anomalies.loc[anomalies['is_anomaly'] == '0']
114/67:
truenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == '0']

falsenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == '1']
114/68:
#Traditional FPR and TPR formmulae
#tpr = truepositives.count()/(truepositives.count() + falsenegatives.count())
#fpr = falsepositives.count()/(falsepositives.count() + truenegatives.count())

#IDEAL Paper based TRP/FPR rates
fpr = falsepositives.count()/n.count()
tpr = truepositives.count()/p.count()
fnr = 1-tpr
tnr = 1-fpr
114/69:
precision = truepositives.count()/(tpr + fpr)
recall = tpr/(tpr + fnr)
114/70: f1 = 2 * ((precision * recall)/(precision + recall))
114/71: print(f1)
114/72: print(p,n)
114/73:
p = df_indexed.loc[df_indexed['is_anomaly'] == 1]

n = df_indexed.loc[df_indexed['is_anomaly'] == 0]
114/74: print(p,n)
114/75: print(truepositives,falsepositives)
114/76:
truepositives = anomalies.loc[anomalies['is_anomaly'] == 1]

falsepositives = anomalies.loc[anomalies['is_anomaly'] == 0]
114/77: print(truepositives,falsepositives)
114/78:
truenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == '0']

falsenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == '1']
114/79:
truenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == 0]

falsenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == 1]
114/80: print(truepositives,falsepositives,sep="\n")
114/81: print(truenegatives,falsenegatives,sep="\n")
114/82: print(fpr,tpr,fnr,tnr,sep = "\n")
114/83: falsepositives.count()
114/84: len(falsepositives)
114/85: len(truepositives)
114/86: len(n)
114/87: len(p)
114/88:
#Traditional FPR and TPR formmulae
#tpr = truepositives.count()/(truepositives.count() + falsenegatives.count())
#fpr = falsepositives.count()/(falsepositives.count() + truenegatives.count())

#IDEAL Paper based TRP/FPR rates
fpr = len(falsepositives)/len(n)
tpr = len(truepositives)/len(p)
fnr = 1-tpr
tnr = 1-fpr
114/89: print(fpr,tpr,fnr,tnr,sep = "\n")
114/90:
precision = len(truepositives)/(tpr + fpr)
recall = tpr/(tpr + fnr)
114/91: print(precision,recall,sep="\n")
114/92: f1 = 2 * ((precision * recall)/(precision + recall))
114/93: print(f1)
114/94:
start_time = datetime.now() 
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 15:
         break
    fname = file.split("/")[5].replace('\\','').split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("./STLoutput/A2Benchmark_" + fname +"add_STL")
    print("\n\n\nA2Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    p = df_indexed.loc[df_indexed['is_anomaly'] == 1]
    
    n = df_indexed.loc[df_indexed['is_anomaly'] == 0]
    
    truepositives = anomalies.loc[anomalies['is_anomaly'] == 1]
    
    falsepositives = anomalies.loc[anomalies['is_anomaly'] == 0]
    
    truenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == 0]
    
    falsenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == 1]
    
    #Traditional FPR and TPR formmulae
    #tpr = truepositives.count()/(truepositives.count() + falsenegatives.count())
    #fpr = falsepositives.count()/(falsepositives.count() + truenegatives.count())
    
    #IDEAL Paper based TRP/FPR rates
    fpr = len(falsepositives)/len(n)
    tpr = len(truepositives)/len(p)
    fnr = 1-tpr
    tnr = 1-fpr
    
    precision = len(truepositives)/(tpr + fpr)
    recall = tpr/(tpr + fnr)
    
    f1 = 2 * ((precision * recall)/(precision + recall))
    
    with open("./STLoutput/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        file.write("\n\n\n Statistics:")
        file.write(f"true positives:{len(truepositives)}")
        file.write(f"false positives:{len(falsepositives)}")
        file.write(f"true negatives:{len(truenegatives)}")
        file.write(f"false negatives:{len(falsenegatives)}")
        file.write(f"FPR:{fpr}")
        file.write(f"TPR:{tpr}")
        file.write(f"FNR:{fnr}")
        file.write(f"TNR:{tnr}")
        
        file.write(f"Precision:{precision}")
        file.write(f"REcall:{recall}")
        file.write(f"F1:{f1}")
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print(f"STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
114/95:
start_time = datetime.now() 
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 15:
         break
    fname = file.split("/")[5].replace('\\','').split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("./STLoutput/A2Benchmark_" + fname +"add_STL")
    print("\n\n\nA2Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    p = df_indexed.loc[df_indexed['is_anomaly'] == 1]
    
    n = df_indexed.loc[df_indexed['is_anomaly'] == 0]
    
    truepositives = anomalies.loc[anomalies['is_anomaly'] == 1]
    
    falsepositives = anomalies.loc[anomalies['is_anomaly'] == 0]
    
    truenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == 0]
    
    falsenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == 1]
    
    #Traditional FPR and TPR formmulae
    #tpr = truepositives.count()/(truepositives.count() + falsenegatives.count())
    #fpr = falsepositives.count()/(falsepositives.count() + truenegatives.count())
    
    #IDEAL Paper based TRP/FPR rates
    fpr = len(falsepositives)/len(n)
    tpr = len(truepositives)/len(p)
    fnr = 1-tpr
    tnr = 1-fpr
    
    precision = len(truepositives)/(tpr + fpr)
    recall = tpr/(tpr + fnr)
    
    f1 = 2 * ((precision * recall)/(precision + recall))
    
    with open("./STLoutput/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        file.write("\n\n\n Statistics:")
        file.write(f"\ntrue positives:{len(truepositives)}")
        file.write(f"\nfalse positives:{len(falsepositives)}")
        file.write(f"\ntrue negatives:{len(truenegatives)}")
        file.write(f"\nfalse negatives:{len(falsenegatives)}")
        file.write(f"\nFPR:{fpr}")
        file.write(f"\nTPR:{tpr}")
        file.write(f"\nFNR:{fnr}")
        file.write(f"\nTNR:{tnr}")
        
        file.write(f"\nPrecision:{precision}")
        file.write(f"\nREcall:{recall}")
        file.write(f"\nF1:{f1}")
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print(f"STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
114/96:
start_time = datetime.now() 
f1_plot = []
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 15:
         break
    fname = file.split("/")[5].replace('\\','').split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("./STLoutput/A2Benchmark_" + fname +"add_STL")
    print("\n\n\nA2Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    p = df_indexed.loc[df_indexed['is_anomaly'] == 1]
    
    n = df_indexed.loc[df_indexed['is_anomaly'] == 0]
    
    truepositives = anomalies.loc[anomalies['is_anomaly'] == 1]
    
    falsepositives = anomalies.loc[anomalies['is_anomaly'] == 0]
    
    truenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == 0]
    
    falsenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == 1]
    
    #Traditional FPR and TPR formmulae
    #tpr = truepositives.count()/(truepositives.count() + falsenegatives.count())
    #fpr = falsepositives.count()/(falsepositives.count() + truenegatives.count())
    
    #IDEAL Paper based TRP/FPR rates
    fpr = len(falsepositives)/len(n)
    tpr = len(truepositives)/len(p)
    fnr = 1-tpr
    tnr = 1-fpr
    
    precision = len(truepositives)/(tpr + fpr)
    recall = tpr/(tpr + fnr)
    
    f1 = 2 * ((precision * recall)/(precision + recall))
    f1plot.append(f1_plot)
    
    with open("./STLoutput/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        file.write("\n\n\n Statistics:")
        file.write(f"\ntrue positives:{len(truepositives)}")
        file.write(f"\nfalse positives:{len(falsepositives)}")
        file.write(f"\ntrue negatives:{len(truenegatives)}")
        file.write(f"\nfalse negatives:{len(falsenegatives)}")
        file.write(f"\nFPR:{fpr}")
        file.write(f"\nTPR:{tpr}")
        file.write(f"\nFNR:{fnr}")
        file.write(f"\nTNR:{tnr}")
        
        file.write(f"\nPrecision:{precision}")
        file.write(f"\nREcall:{recall}")
        file.write(f"\nF1:{f1}")
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print(f"STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
114/97:
start_time = datetime.now() 
f1_plot = []
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 15:
         break
    fname = file.split("/")[5].replace('\\','').split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("./STLoutput/A2Benchmark_" + fname +"add_STL")
    print("\n\n\nA2Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    p = df_indexed.loc[df_indexed['is_anomaly'] == 1]
    
    n = df_indexed.loc[df_indexed['is_anomaly'] == 0]
    
    truepositives = anomalies.loc[anomalies['is_anomaly'] == 1]
    
    falsepositives = anomalies.loc[anomalies['is_anomaly'] == 0]
    
    truenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == 0]
    
    falsenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == 1]
    
    #Traditional FPR and TPR formmulae
    #tpr = truepositives.count()/(truepositives.count() + falsenegatives.count())
    #fpr = falsepositives.count()/(falsepositives.count() + truenegatives.count())
    
    #IDEAL Paper based TRP/FPR rates
    fpr = len(falsepositives)/len(n)
    tpr = len(truepositives)/len(p)
    fnr = 1-tpr
    tnr = 1-fpr
    
    precision = len(truepositives)/(tpr + fpr)
    recall = tpr/(tpr + fnr)
    
    f1 = 2 * ((precision * recall)/(precision + recall))
    f1_plot.append(f1)
    
    with open("./STLoutput/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        file.write("\n\n\n Statistics:")
        file.write(f"\ntrue positives:{len(truepositives)}")
        file.write(f"\nfalse positives:{len(falsepositives)}")
        file.write(f"\ntrue negatives:{len(truenegatives)}")
        file.write(f"\nfalse negatives:{len(falsenegatives)}")
        file.write(f"\nFPR:{fpr}")
        file.write(f"\nTPR:{tpr}")
        file.write(f"\nFNR:{fnr}")
        file.write(f"\nTNR:{tnr}")
        
        file.write(f"\nPrecision:{precision}")
        file.write(f"\nREcall:{recall}")
        file.write(f"\nF1:{f1}")
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print(f"STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
plt.plot(f1_plot)
114/98:
class TSFeatures:
    
    def __repr__():
        return f"TSFeatures object with properties:{this.mean}, {this.variance}"
114/99: ts1 = new TSFeatures()
114/100: ts1 = TSFeatures()
114/101:
ts1.mean = features[0]
ts1.variance = features[1]
114/102: ts1
114/103:
class TSFeatures:
    
    def __repr__(self):
        return f"TSFeatures object with properties:{this.mean}, {this.variance}"
114/104: ts1 = TSFeatures()
114/105:
ts1.mean = features[0]
ts1.variance = features[1]
114/106: ts1
114/107:
class TSFeatures:
    
    def __repr__(self):
        return f"TSFeatures object with properties:{self.mean}, {self.variance}"
114/108: ts1 = TSFeatures()
114/109:
ts1.mean = features[0]
ts1.variance = features[1]
114/110: ts1
114/111:
features = extract_features(df_indexed[0])
print(features)
114/112:
features = extract_features(df_indexed['value'])
print(features)
114/113:
features = extract_features(df_indexed)
print(features)
114/114: features[1,]
114/115: features[1]
114/116: len(features)
114/117: features
114/118: features[0]
114/119:
tsfeatures1  = pandas2ri.ri2py(features)
tsfeatures1
114/120: type(pandas2ri.ri2py(features))
114/121: tsfeatures1[0]
114/122: tsfeatures1[0,0]
112/45: runfile('D:/Temp/time-series/STL Decomposition for Time Series Anomalous Record Detection.py', wdir='D:/Temp/time-series')
112/46: runfile('D:/Temp/time-series/STL Decomposition for Time Series Anomalous Record Detection.py', wdir='D:/Temp/time-series')
112/47: runfile('D:/Temp/time-series/STL Decomposition for Time Series Anomalous Record Detection.py', wdir='D:/Temp/time-series')
114/123:
from enum import enum
    class TSFeatures(enum):
        MEAN = 0
        VARIANCE = 1
        LUMPINESS = 2
        LSHIFT = 3
        VCHANGE = 4
        LINEARITY = 5
        CURVATURE = 6
        SPIKINESS = 7
        
        def __repr__(self):
            return f"TSFeatures object with properties:{self.mean}, {self.variance}"
114/124:
from enum import enum
class TSFeatures(enum):
    MEAN = 0
    VARIANCE = 1
    LUMPINESS = 2
    LSHIFT = 3
    VCHANGE = 4
    LINEARITY = 5
    CURVATURE = 6
    SPIKINESS = 7
    SEAD
    
    def __repr__(self):
        return f"TSFeatures object with properties:{self.mean}, {self.variance}"
114/125:
import enum
class TSFeatures(enum):
    MEAN = 0
    VARIANCE = 1
    LUMPINESS = 2
    LSHIFT = 3
    VCHANGE = 4
    LINEARITY = 5
    CURVATURE = 6
    SPIKINESS = 7
    SEAD
    
    def __repr__(self):
        return f"TSFeatures object with properties:{self.mean}, {self.variance}"
114/126:
import enum
class TSFeatures(enum):
    MEAN = 0
    VARIANCE = 1
    LUMPINESS = 2
    LSHIFT = 3
    VCHANGE = 4
    LINEARITY = 5
    CURVATURE = 6
    SPIKINESS = 7

    
    def __repr__(self):
        return f"TSFeatures object with properties:{self.mean}, {self.variance}"
114/127:
import enum
class TSFeatures(Enum):
    MEAN = 0
    VARIANCE = 1
    LUMPINESS = 2
    LSHIFT = 3
    VCHANGE = 4
    LINEARITY = 5
    CURVATURE = 6
    SPIKINESS = 7

    
    def __repr__(self):
        return f"TSFeatures object with properties:{self.mean}, {self.variance}"
114/128:
from enum import Enum
class TSFeatures(Enum):
    MEAN = 0
    VARIANCE = 1
    LUMPINESS = 2
    LSHIFT = 3
    VCHANGE = 4
    LINEARITY = 5
    CURVATURE = 6
    SPIKINESS = 7

    
    def __repr__(self):
        return f"TSFeatures object with properties:{self.mean}, {self.variance}"
114/129: ts1 = TSFeatures()
114/130:
ts1.mean = features[0]
ts1.variance = features[1]
114/131: ts1
114/132: tsfeatures1[TSFeatures.SPIKINESS]
114/133: TSFeatures.SPIKINESS
114/134:
from enum import Enum
class TSFeatures(Enum):
    MEAN = 0
    VARIANCE = 1
    LUMPINESS = 2
    LSHIFT = 3
    VCHANGE = 4
    LINEARITY = 5
    CURVATURE = 6
    SPIKINESS = 7

    
    def __repr__(self):
        return f"TSFeatures object with properties:{self.mean}, {self.variance}"
114/135: TSFeatures.SPIKINESS
114/136: print(TSFeatures.SPIKINESS)
114/137: print(repr(TSFeatures.SPIKINESS))
114/138:
from enum import Enum
class TSFeatures(Enum):
    MEAN = 0
    VARIANCE = 1
    LUMPINESS = 2
    LSHIFT = 3
    VCHANGE = 4
    LINEARITY = 5
    CURVATURE = 6
    SPIKINESS = 7

    
    def __repr__(self):
        return f"TSFeatures object with properties:{self.mean}, {self.variance}"
114/139:
from enum import Enum
class TSFeatures(Enum):
    MEAN = 0
    VARIANCE = 1
    LUMPINESS = 2
    LSHIFT = 3
    VCHANGE = 4
    LINEARITY = 5
    CURVATURE = 6
    SPIKINESS = 7
114/140: ts1 = TSFeatures()
114/141:
ts1.mean = features[0]
ts1.variance = features[1]
114/142: ts1
114/143: tsfeatures1[TSFeatures.SPIKINESS]
114/144: print(TSFeatures.SPIKINESS)
114/145: print(repr(TSFeatures.SPIKINESS))
116/1: tsfeatures1[TSFeatures.SPIKINESS]
117/1:
import pandas as pd
from matplotlib import pyplot as plt
import os
from statsmodels.tsa.seasonal import seasonal_decompose
import glob
from datetime import datetime
117/2:
start_time = datetime.now()
all_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/**/*.csv', recursive=True)
end_time = datetime.now()
print(f'Loaded the paths of {len(all_csv)} files from disk. Took {end_time-start_time}')
117/3:
df= pd.read_csv(all_csv[0])
df
117/4: df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
117/5: df
117/6:
import pandas as pd
from matplotlib import pyplot as plt
import os
from statsmodels.api as sm
import glob
from datetime import datetime
117/7:
import pandas as pd
from matplotlib import pyplot as plt
import os
import statsmodels.api as sm
import glob
from datetime import datetime
117/8:
X = sm.add_constant(df['value'])
model = sm.OLS(df['value'],X)
results = model.fit()
117/9: plt.scatter(df['value'],aplha=0.3)
117/10: plt.scatter(df['value'],df['time'],alpha=0.3)
117/11: plt.scatter(df['value'],df['timestamp'],alpha=0.3)
117/12:
from pandas.plotting import autocorrelation_plot
# Draw Plot
plt.rcParams.update({'figure.figsize':(9,5), 'figure.dpi':120})
autocorrelation_plot(df.value.tolist())
117/13:
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Calculate ACF and PACF upto 50 lags
# acf_50 = acf(df.value, nlags=50)
# pacf_50 = pacf(df.value, nlags=50)

# Draw Plot
fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(df.value.tolist(), lags=50, ax=axes[0])
plot_pacf(df.value.tolist(), lags=50, ax=axes[1])
117/14:
from pandas.plotting import lag_plot

# Plot
fig, axes = plt.subplots(1, 4, figsize=(10,3), sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(df.value, lag=i+1, ax=ax, c='firebrick')
    ax.set_title('Lag ' + str(i+1))

fig.suptitle('Lag Plots of Sun Spots Area \n(Points get wide and scattered with increasing lag -> lesser correlation)\n', y=1.15)
117/15:
from statsmodels.nonparametric.smoothers_lowess import lowess

plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

# 1. Moving Average
df_ma = df.value.rolling(3, center=True, closed='both').mean()

# 2. Loess Smoothing (5% and 15%)
df_loess_5 = pd.DataFrame(lowess(df.value, np.arange(len(df_orig.value)), frac=0.05)[:, 1], index=df_orig.index, columns=['value'])
df_loess_15 = pd.DataFrame(lowess(df.value, np.arange(len(df_orig.value)), frac=0.15)[:, 1], index=df_orig.index, columns=['value'])

# Plot
fig, axes = plt.subplots(4,1, figsize=(7, 7), sharex=True, dpi=120)
df_orig['value'].plot(ax=axes[0], color='k', title='Original Series')
df_loess_5['value'].plot(ax=axes[1], title='Loess Smoothed 5%')
df_loess_15['value'].plot(ax=axes[2], title='Loess Smoothed 15%')
df_ma.plot(ax=axes[3], title='Moving Average (3)')
fig.suptitle('How to Smoothen a Time Series', y=0.95, fontsize=14)
plt.show()
117/16:
from statsmodels.nonparametric.smoothers_lowess import lowess

plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

# 1. Moving Average
df_ma = df.value.rolling(3, center=True, closed='timestamp').mean()

# 2. Loess Smoothing (5% and 15%)
df_loess_5 = pd.DataFrame(lowess(df.value, np.arange(len(df_orig.value)), frac=0.05)[:, 1], index=df_orig.index, columns=['value'])
df_loess_15 = pd.DataFrame(lowess(df.value, np.arange(len(df_orig.value)), frac=0.15)[:, 1], index=df_orig.index, columns=['value'])

# Plot
fig, axes = plt.subplots(4,1, figsize=(7, 7), sharex=True, dpi=120)
df_orig['value'].plot(ax=axes[0], color='k', title='Original Series')
df_loess_5['value'].plot(ax=axes[1], title='Loess Smoothed 5%')
df_loess_15['value'].plot(ax=axes[2], title='Loess Smoothed 15%')
df_ma.plot(ax=axes[3], title='Moving Average (3)')
fig.suptitle('How to Smoothen a Time Series', y=0.95, fontsize=14)
plt.show()
117/17:
from statsmodels.nonparametric.smoothers_lowess import lowess

plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

# 1. Moving Average
df_ma = df.value.rolling(3, center=True).mean()

# 2. Loess Smoothing (5% and 15%)
df_loess_5 = pd.DataFrame(lowess(df.value, np.arange(len(df_orig.value)), frac=0.05)[:, 1], index=df_orig.index, columns=['value'])
df_loess_15 = pd.DataFrame(lowess(df.value, np.arange(len(df_orig.value)), frac=0.15)[:, 1], index=df_orig.index, columns=['value'])

# Plot
fig, axes = plt.subplots(4,1, figsize=(7, 7), sharex=True, dpi=120)
df_orig['value'].plot(ax=axes[0], color='k', title='Original Series')
df_loess_5['value'].plot(ax=axes[1], title='Loess Smoothed 5%')
df_loess_15['value'].plot(ax=axes[2], title='Loess Smoothed 15%')
df_ma.plot(ax=axes[3], title='Moving Average (3)')
fig.suptitle('How to Smoothen a Time Series', y=0.95, fontsize=14)
plt.show()
117/18:
import pandas as pd
from matplotlib import pyplot as plt
import os
import statsmodels.api as sm
import glob
from datetime import datetime
import numpy as np
117/19:
from statsmodels.nonparametric.smoothers_lowess import lowess

plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

# 1. Moving Average
df_ma = df.value.rolling(3, center=True).mean()

# 2. Loess Smoothing (5% and 15%)
df_loess_5 = pd.DataFrame(lowess(df.value, np.arange(len(df.value)), frac=0.05)[:, 1], index=df.index, columns=['value'])
df_loess_15 = pd.DataFrame(lowess(df.value, np.arange(len(df.value)), frac=0.15)[:, 1], index=df.index, columns=['value'])

# Plot
fig, axes = plt.subplots(4,1, figsize=(7, 7), sharex=True, dpi=120)
df_orig['value'].plot(ax=axes[0], color='k', title='Original Series')
df_loess_5['value'].plot(ax=axes[1], title='Loess Smoothed 5%')
df_loess_15['value'].plot(ax=axes[2], title='Loess Smoothed 15%')
df_ma.plot(ax=axes[3], title='Moving Average (3)')
fig.suptitle('How to Smoothen a Time Series', y=0.95, fontsize=14)
plt.show()
117/20:
from statsmodels.nonparametric.smoothers_lowess import lowess

plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

# 1. Moving Average
df_ma = df.value.rolling(3, center=True).mean()

# 2. Loess Smoothing (5% and 15%)
df_loess_5 = pd.DataFrame(lowess(df.value, np.arange(len(df.value)), frac=0.05)[:, 1], index=df.index, columns=['value'])
df_loess_15 = pd.DataFrame(lowess(df.value, np.arange(len(df.value)), frac=0.15)[:, 1], index=df.index, columns=['value'])

# Plot
fig, axes = plt.subplots(4,1, figsize=(7, 7), sharex=True, dpi=120)
df['value'].plot(ax=axes[0], color='k', title='Original Series')
df_loess_5['value'].plot(ax=axes[1], title='Loess Smoothed 5%')
df_loess_15['value'].plot(ax=axes[2], title='Loess Smoothed 15%')
df_ma.plot(ax=axes[3], title='Moving Average (3)')
fig.suptitle('How to Smoothen a Time Series', y=0.95, fontsize=14)
plt.show()
117/21:
from statsmodels.nonparametric.smoothers_lowess import lowess

plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

# 1. Moving Average
df_ma = df.value.rolling(5, center=True).mean()

# 2. Loess Smoothing (5% and 15%)
df_loess_5 = pd.DataFrame(lowess(df.value, np.arange(len(df.value)), frac=0.05)[:, 1], index=df.index, columns=['value'])
df_loess_15 = pd.DataFrame(lowess(df.value, np.arange(len(df.value)), frac=0.15)[:, 1], index=df.index, columns=['value'])

# Plot
fig, axes = plt.subplots(4,1, figsize=(7, 7), sharex=True, dpi=120)
df['value'].plot(ax=axes[0], color='k', title='Original Series')
df_loess_5['value'].plot(ax=axes[1], title='Loess Smoothed 5%')
df_loess_15['value'].plot(ax=axes[2], title='Loess Smoothed 15%')
df_ma.plot(ax=axes[3], title='Moving Average (3)')
fig.suptitle('How to Smoothen a Time Series', y=0.95, fontsize=14)
plt.show()
117/22:
from statsmodels.nonparametric.smoothers_lowess import lowess

plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

# 1. Moving Average
df_ma = df.value.rolling(10, center=True).mean()

# 2. Loess Smoothing (5% and 15%)
df_loess_5 = pd.DataFrame(lowess(df.value, np.arange(len(df.value)), frac=0.05)[:, 1], index=df.index, columns=['value'])
df_loess_15 = pd.DataFrame(lowess(df.value, np.arange(len(df.value)), frac=0.15)[:, 1], index=df.index, columns=['value'])

# Plot
fig, axes = plt.subplots(4,1, figsize=(7, 7), sharex=True, dpi=120)
df['value'].plot(ax=axes[0], color='k', title='Original Series')
df_loess_5['value'].plot(ax=axes[1], title='Loess Smoothed 5%')
df_loess_15['value'].plot(ax=axes[2], title='Loess Smoothed 15%')
df_ma.plot(ax=axes[3], title='Moving Average (3)')
fig.suptitle('How to Smoothen a Time Series', y=0.95, fontsize=14)
plt.show()
117/23:
from statsmodels.nonparametric.smoothers_lowess import lowess

plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

# 1. Moving Average
df_ma = df.value.rolling(50, center=True).mean()

# 2. Loess Smoothing (5% and 15%)
df_loess_5 = pd.DataFrame(lowess(df.value, np.arange(len(df.value)), frac=0.05)[:, 1], index=df.index, columns=['value'])
df_loess_15 = pd.DataFrame(lowess(df.value, np.arange(len(df.value)), frac=0.15)[:, 1], index=df.index, columns=['value'])

# Plot
fig, axes = plt.subplots(4,1, figsize=(7, 7), sharex=True, dpi=120)
df['value'].plot(ax=axes[0], color='k', title='Original Series')
df_loess_5['value'].plot(ax=axes[1], title='Loess Smoothed 5%')
df_loess_15['value'].plot(ax=axes[2], title='Loess Smoothed 15%')
df_ma.plot(ax=axes[3], title='Moving Average (3)')
fig.suptitle('How to Smoothen a Time Series', y=0.95, fontsize=14)
plt.show()
118/1:
import pandas as pd
from matplotlib import pyplot as plt
import os
import statsmodels.api as sm
import glob
from datetime import datetime
import numpy as np
118/2:
import pandas as pd
from matplotlib import pyplot as plt
import os
import statsmodels.api as sm
import glob
from datetime import datetime as dt
import numpy as np
118/3:
start_time = dt.now()
all_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/**/*.csv', recursive=True)
end_time = dt.now()
print(f'Loaded the paths of {len(all_csv)} files from disk. Took {end_time-start_time}')
118/4:
df= pd.read_csv(all_csv[0])
df
118/5: df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
118/6: df
118/7:
plt.tight_layout()
plt.show()
119/1: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
119/2: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
119/3: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
119/4: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
119/5: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
119/6: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
119/7: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
119/8: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
119/9: plt.gcf()
121/1: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
121/2: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
121/3: df['value']
121/4: fft(df['value'])
121/5: fft(df['value'].tolist())
121/6: len(df['value'])
121/7: len(fft(df['value'].tolist()))
121/8: len(fft(df['value'].tolist(),nfft))
121/9: df['timestamp']
121/10: df['timestamp'].tolist()
121/11: df['timestamp'].tolist()[0]
121/12: type(df['timestamp'].tolist()[0])
121/13: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
121/14: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
121/15: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
121/16: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
121/17: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
121/18: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
121/19: df
121/20: df.index = df['timestamp']
121/21: df
118/8: df
118/9: df
118/10: df.shape
118/11:
data = df.copy()
size=3
shape = data.shape[:-1] + (data.shape[-1] - block + 1, block)
118/12:
data = df.copy()
size=3
shape = data.shape[:-1] + (data.shape[-1] - size + 1, size)
print(shape)
118/13:
data = df.copy()
size=3
shape = data.shape[:-1] + (data.shape[-1] - size + 1, size)
print(shape)
strides = data.strides + (data.strides[-1],)
print(strides)
118/14:
##FIRST DESIGN A SLIDING WINDOW ON THE DATA
def rolling_window(data,size):
    window_arr = [data[i:i+size] for i in range(len(array) - 2)]
    return window_arr
118/15: print(rolling_window(df['values']))
118/16: print(rolling_window(df['value']))
118/17: print(rolling_window(df['value'],5))
118/18:
##FIRST DESIGN A SLIDING WINDOW ON THE DATA
def rolling_window(data,size):
    window_arr = [data[i:i+size] for i in range(len(data) - 2)]
    return window_arr
118/19: print(rolling_window(df['value'],5))
118/20: df.rolling(5)
118/21: print(df.rolling(5))
118/22: print(df.rolling(5)[0])
118/23: print([i for i in df.rolling(5)])
118/24:
def movingaverage(arr,blocksize=10):
    avg = []
    for i in rolling_window(arr,blocksize):
        avg = i.mean()
    print(avg)
    
movingaverage(df['value'].tolist())
118/25: df['value'].rolling(10).mean()
118/26: df['value'].rolling(5).mean()
121/22: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
121/23: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
121/24: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
121/25: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
121/26: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
121/27: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
121/28: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
121/29: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
121/30: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
121/31: runfile('D:/Temp/time-series/plotting and fft.py', wdir='D:/Temp/time-series')
121/32: ro.r(f'install.packages("devtools")')
121/33: ro.r(f'devtools::install_github("robjhyndman/anomalous")')
124/1:
def movingaverage(arr,blocksize=10):
    avg = []
    for i in rolling_window(arr,blocksize):
        avg = i.mean()
    print(avg)
    
movingaverage(df['value'].tolist())
124/2:
import pandas as pd
from matplotlib import pyplot as plt
import os
import statsmodels.api as sm
import glob
from datetime import datetime as dt
import numpy as np
124/3:
start_time = dt.now()
all_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/**/*.csv', recursive=True)
end_time = dt.now()
print(f'Loaded the paths of {len(all_csv)} files from disk. Took {end_time-start_time}')
124/4:
df= pd.read_csv(all_csv[0])
df
124/5: df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
124/6: df
124/7:
X = sm.add_constant(df['value'])
model = sm.OLS(df['value'],X)
results = model.fit()
124/8: plt.scatter(df['value'],df['timestamp'],alpha=0.3)
124/9:
from pandas.plotting import autocorrelation_plot
# Draw Plot
plt.rcParams.update({'figure.figsize':(9,5), 'figure.dpi':120})
autocorrelation_plot(df.value.tolist())
124/10:
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Calculate ACF and PACF upto 50 lags
# acf_50 = acf(df.value, nlags=50)
# pacf_50 = pacf(df.value, nlags=50)

# Draw Plot
fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(df.value.tolist(), lags=50, ax=axes[0])
plot_pacf(df.value.tolist(), lags=50, ax=axes[1])
124/11:
from pandas.plotting import lag_plot

# Plot
fig, axes = plt.subplots(1, 4, figsize=(10,3), sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(df.value, lag=i+1, ax=ax, c='firebrick')
    ax.set_title('Lag ' + str(i+1))

fig.suptitle('Lag Plots of Sun Spots Area \n(Points get wide and scattered with increasing lag -> lesser correlation)\n', y=1.15)
124/12:
from statsmodels.nonparametric.smoothers_lowess import lowess

plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

# 1. Moving Average
df_ma = df.value.rolling(50, center=True).mean()

# 2. Loess Smoothing (5% and 15%)
df_loess_5 = pd.DataFrame(lowess(df.value, np.arange(len(df.value)), frac=0.05)[:, 1], index=df.index, columns=['value'])
df_loess_15 = pd.DataFrame(lowess(df.value, np.arange(len(df.value)), frac=0.15)[:, 1], index=df.index, columns=['value'])

# Plot
fig, axes = plt.subplots(4,1, figsize=(7, 7), sharex=True, dpi=120)
df['value'].plot(ax=axes[0], color='k', title='Original Series')
df_loess_5['value'].plot(ax=axes[1], title='Loess Smoothed 5%')
df_loess_15['value'].plot(ax=axes[2], title='Loess Smoothed 15%')
df_ma.plot(ax=axes[3], title='Moving Average (3)')
fig.suptitle('How to Smoothen a Time Series', y=0.95, fontsize=14)
plt.show()
124/13:
##FIRST DESIGN A SLIDING WINDOW ON THE DATA
def rolling_window(data,size):
    window_arr = [data[i:i+size] for i in range(len(data) - 2)]
    return window_arr
124/14: print(rolling_window(df['value'],5))
124/15:
def movingaverage(arr,blocksize=10):
    avg = []
    for i in rolling_window(arr,blocksize):
        avg = i.mean()
    print(avg)
    
movingaverage(df['value'].tolist())
124/16:
import pandas as pd
from matplotlib import pyplot as plt
import os
import statsmodels.api as sm
import glob
from datetime import datetime as dt
import numpy as np
124/17:
start_time = dt.now()
all_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/**/*.csv', recursive=True)
end_time = dt.now()
print(f'Loaded the paths of {len(all_csv)} files from disk. Took {end_time-start_time}')
124/18:
df= pd.read_csv(all_csv[0])
df
124/19: df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
124/20: df
124/21:
X = sm.add_constant(df['value'])
model = sm.OLS(df['value'],X)
results = model.fit()
124/22:
##FIRST DESIGN A SLIDING WINDOW ON THE DATA
def rolling_window(data,size):
    window_arr = [data[i:i+size] for i in range(len(data) - 2)]
    return window_arr
124/23: print(rolling_window(df['value'],5))
124/24:
def movingaverage(arr,blocksize=10):
    avg = []
    for i in rolling_window(arr,blocksize):
        avg = i.mean()
    print(avg)
    
movingaverage(df['value'].tolist())
124/25: print(rolling_window(df['value'],5)[0])
124/26: print(rolling_window(df['value'],5))
124/27: print(rolling_window(df['value'],5)[0])
124/28: print(rolling_window(df['value'].tolist(),5)[0])
124/29:
def movingaverage(arr,blocksize=10):
    avg = []
    for i in rolling_window(arr,blocksize):
        avg = i.mean()
    print(avg)
    
movingaverage(df['value'])
124/30: df['value'].rolling(5).mean()
124/31: df['value'].rolling(5).mean().tolist()
126/1:
import os
os.getcwd()
126/2:
import pandas as pd
import matplotlib.pyplot as plt
series = read_csv('./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
126/3:
import pandas as pd
import matplotlib.pyplot as plt
series = pd.read_csv('./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
126/4: series.head()
126/5: series.plot()
126/6: series.plot('value')
126/7:
series['timestamp'] = pd.to_datetime(series['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
plt.plot_date(series['timestamp'],series['value'])
126/8:
series['timestamp'] = pd.to_datetime(series['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
plt.plot_date(series['timestamp'],series['value'])
plt.gcf().autofmt_xdate()
126/9: plt.lag_plot(series)
126/10: pd.plotting.lag_plot(series)
127/1: runfile('D:/Temp/time-series/untitled0.py', wdir='D:/Temp/time-series')
127/2: runfile('D:/Temp/time-series/untitled0.py', wdir='D:/Temp/time-series')
127/3: series['timestamp']
126/11:
series.index = series['timestamp']
pd.plotting.lag_plot(series)
127/4: runfile('D:/Temp/time-series/untitled0.py', wdir='D:/Temp/time-series')
127/5: series
127/6: series.set_index('timestamp')
127/7: runfile('D:/Temp/time-series/untitled0.py', wdir='D:/Temp/time-series')
127/8: series.set_index('timestamp')
127/9: pd.plotting.lag_plot(series)
127/10: runfile('D:/Temp/time-series/untitled0.py', wdir='D:/Temp/time-series')
127/11: runfile('D:/Temp/time-series/untitled0.py', wdir='D:/Temp/time-series')
127/12: runfile('D:/Temp/time-series/untitled0.py', wdir='D:/Temp/time-series')
126/12: series.set_index('timestamp')
126/13:
values = pd.DataFrame(series.value)
df = pd.concat([values.shift(1),values],axis =1)
df.columns = ['t-1','t']
result = df.corr()
print(result)
126/14:
for lag in range(0,6):
    df = pd.concat([values.shift(lag),values],axis =1)
    df.columns = ['t-'+str(lag),'t']
    result = df.corr()
    print(result)
126/15:
for lag in range(0,10):
    df = pd.concat([values.shift(lag),values],axis =1)
    df.columns = ['t-'+str(lag),'t']
    result = df.corr()
    print(result)
126/16:
for lag in range(0,10):
    print("Lag : " + str(lag) + "\n")
    df = pd.concat([values.shift(lag),values],axis =1)
    df.columns = ['t-'+str(lag),'t']
    result = df.corr()
    print(result)
    print("\n\n\n")
126/17:
for lag in range(0,10):
    print("Lag : " + str(lag) + "\n")
    df = pd.concat([values.shift(lag),values],axis =1)
    df.columns = ['t-'+str(lag),'t']
    result = df.corr()
    print(result)
    print("\n\n")
126/18:
for lag in range(0,20):
    print("Lag : " + str(lag) + "\n")
    df = pd.concat([values.shift(lag),values],axis =1)
    df.columns = ['t-'+str(lag),'t']
    result = df.corr()
    print(result)
    print("\n\n")
126/19: result
126/20: result[0][1]
126/21: result[0]
126/22: result
126/23: result.t[0]
126/24:
autocorr = []
for lag in range(0,20):
    print("Lag : " + str(lag) + "\n")
    df = pd.concat([values.shift(lag),values],axis =1)
    df.columns = ['t-'+str(lag),'t']
    result = df.corr()
    print(result)
    print("\n\n")
    autocorr += result.t[0]
126/25: plt.plot(autocorr)
126/26: plt.plot(enumerate(autocorr))
126/27: plt.plot([i for i in enumerate(autocorr)])
126/28: autocorr
126/29:
autocorr = []
for lag in range(0,20):
    print("Lag : " + str(lag) + "\n")
    df = pd.concat([values.shift(lag),values],axis =1)
    df.columns = ['t-'+str(lag),'t']
    result = df.corr()
    autocorr += result.t[0]
    print(result)
    print("\n\n")
126/30: autocorr
126/31: result
126/32: result.t[0]
126/33:
autocorr = []
for lag in range(0,20):
    print("Lag : " + str(lag) + "\n")
    df = pd.concat([values.shift(lag),values],axis =1)
    df.columns = ['t-'+str(lag),'t']
    result = df.corr()
    autocorr += result.t[0]
    print(result.t[0])
    print("\n\n")
126/34:
autocorr = []
for lag in range(0,20):
    print("Lag : " + str(lag) + "\n")
    df = pd.concat([values.shift(lag),values],axis =1)
    df.columns = ['t-'+str(lag),'t']
    result = df.corr()
    autocorr.append(result.t[0])
    print(result.t[0])
    print("\n\n")
126/35: plt.plot([i for i in enumerate(autocorr)])
126/36: plt.plot([i for i in (autocorr)])
126/37:
autocorr = []
for lag in range(0,20):
    print("Lag : " + str(lag) + "\n")
    df = pd.concat([values.shift(lag),values],axis =1)
    df.columns = ['t-'+str(lag),'t']
    result = df.corr()
    autocorr.append(result.t[0])
    print(result)
    print("\n\n")
126/38:
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(series)
126/39:
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series, lags=31)
126/40:
from statsmodels.graphics.tsaplots import plot_acf
series['timestamp'] = str(series['timestamp'])
plot_acf(series, lags=31)
126/41:
from statsmodels.graphics.tsaplots import plot_acf
series = pd.read_csv('./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
plot_acf(series, lags=31)
126/42:
from statsmodels.graphics.tsaplots import plot_acf
series = pd.read_csv('./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
plot_acf(series[:1], lags=31)
126/43:
from statsmodels.graphics.tsaplots import plot_acf
series = pd.read_csv('./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
plot_acf(series[1], lags=31)
126/44:
from statsmodels.graphics.tsaplots import plot_acf
series = pd.read_csv('./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
plot_acf(series['value'], lags=31)
126/45:
from pandas.plotting import autocorrelation_plot
series = pd.read_csv('./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
autocorrelation_plot(series)
126/46:
from pandas.plotting import lag_plot
lag_plot(series)
126/47:
from pandas.plotting import lag_plot
series = pd.read_csv('./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
lag_plot(series)
126/48: from sklearn.metrics import mean_squared_error
126/49:
# create lagged dataset
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
126/50:
# create lagged dataset
values = pd.DataFrame(series.values)
dataframe = pd.concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
126/51:
# create lagged dataset
values = pd.DataFrame(series.values)
dataframe = pd.concat([values.shift(1), values], axis=1)
#dataframe.columns = ['t-1', 't+1']
dataframe
126/52: series
126/53:
# create lagged dataset
values = pd.DataFrame(series.value)
dataframe = pd.concat([values.shift(1), values], axis=1)
#dataframe.columns = ['t-1', 't+1']
dataframe
126/54:
# create lagged dataset
values = pd.DataFrame(series.value)
dataframe = pd.concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
dataframe
126/55:
# split into train and test sets
X = dataframe.values
train, test = X[1:len(X)-7], X[len(X)-7:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
126/56:
# persistence model
def model_persistence(x):
    return x
126/57:
# walk-forward validation
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)
126/58:
# walk-forward validation
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)
126/59:
# plot predictions vs expected
pyplot.plot(test_y)
pyplot.plot(predictions, color='red')
pyplot.show()
126/60:
# plot predictions vs expected
plt.plot(test_y)
plt.plot(predictions, color='red')
plt.show()
126/61:
from statsmodels.tsa.ar_model import AutoReg
from math import sqrt
126/62: series = pd.read_csv('./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
126/63:
# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]
126/64:
# train autoregression
model = AutoReg(train, lags=29)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)
126/65:
# split dataset
X = series.value
train, test = X[1:len(X)-7], X[len(X)-7:]
126/66:
# train autoregression
model = AutoReg(train, lags=29)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)
126/67:
X = series.value
train, test = X[1: len(X) - 7], X[len(X) - 7: ]
126/68:
model = AutoReg(train, lag=31)
model_fit = model.fit()
print(f'Coefficients: {model_fit.params}')
126/69:
model = AutoReg(train, lags=31)
model_fit = model.fit()
print(f'Coefficients: {model_fit.params}')
126/70:
predictions = model_fit.predict(start = len(train),end = len(train) + len(test) - 1, dynamic = False)
predictions
126/71:
for i in range(len(predictions)):
    print(f"predictions: {predictions[i]}, Expected: {test[i]}")
126/72:
for i in range(len(predictions)):
    print(f'predictions: {predictions[i]}, Expected: {test[i]}')
126/73:
for i in range(len(predictions)):
    print(f'predictions: {predictions[i]}')
126/74: predictions[0]
126/75: type(predictions)
126/76:
type(predictions)
len(predictions)
126/77:
for i in range(len(predictions):
    print(f'predictions: {predictions.tolist()[i]}, Expected: {test[i]}')
126/78:
for i in range(len(predictions)):
    print(f'predictions: {predictions.tolist()[i]}, Expected: {test[i]}')
126/79:
type(predictions)
len(predictions)
predictions.tolist()
126/80:
for i in range(len(predictions)):
    print(f'predictions: {predictions.tolist()[i]}, Expected: {test.tolist()[i]}')
126/81:
rmse = sqrt(mean_squared_error(test,predictions))
print(f"Test RMSE: {rmse}")
126/82:
model = AutoReg(train, lags=29)
model_fit = model.fit()
print(f'Coefficients: {model_fit.params}')
126/83:
predictions = model_fit.predict(start = len(train),end = len(train) + len(test) - 1, dynamic = False)
predictions
126/84:
type(predictions)
len(predictions)
predictions.tolist()
126/85:
for i in range(len(predictions)):
    print(f'predictions: {predictions.tolist()[i]}, Expected: {test.tolist()[i]}')
126/86:
rmse = sqrt(mean_squared_error(test,predictions))
print(f"Test RMSE: {rmse}")
126/87:
#clean the predictions
predictions = predictions.dropna()
predictions
126/88:
for i in range(len(predictions)):
    print(f'predictions: {predictions.tolist()[i]}, Expected: {test.tolist()[i]}')
126/89:
rmse = sqrt(mean_squared_error(test,predictions))
print(f"Test RMSE: {rmse}")
126/90:
rmse = sqrt(mean_squared_error(test[:len(predictions)],predictions))
print(f"Test RMSE: {rmse}")
126/91:
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
127/13: from statsmodels.tsa.ar_model import AutoReg
127/14: model = AutoReg(series, lags = 100)
127/15: model = AutoReg(series.value, lags = 100)
127/16: model_fit = model.fit()
127/17: coefficients = model_fit.params
127/18: coefficients
127/19: pred = model_fit.predict(start=len(train), end=len(train) + 5,dynamic=False)
127/20: pred = model_fit.predict(start=len(series.value), end=len(series.value) + 5,dynamic=False)
127/21: pred
127/22: len(pred)
127/23: plt.plot(series.value)
127/24: plt.
127/25: plt.plot(pred, color='red')
127/26: plt.show()
127/27: plt.plot(series.value,pred, color='red')
127/28: plt.plot(y=[series.value,pred], color='red')
127/29: plt.gca()
127/30: ax = plt.gca()
127/31: ax
126/92: series = pd.read_csv('./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
126/93:
X = series.value
train, test = X[1: len(X) - 7], X[len(X) - 7: ]
126/94:
window = 29
model = AutoReg(train, lags=29)
coef = model.params
126/95:
window = 29
model = AutoReg(train, lags=29)
model_fit = model.fit()
coef = model_fit.params
126/96:
history = train[len(train) - window:]
history = [history[i] for i in range(len(history))]
predictions = list()
history
126/97:
X = series.value
train, test = X[1: len(X) - 7], X[len(X) - 7: ]
train, test
126/98:
train = train.tolist()
test = test.tolist()
126/99:
window = 29
model = AutoReg(train, lags=29)
model_fit = model.fit()
coef = model_fit.params
126/100:
history = train[len(train) - window:]
history = [history[i] for i in range(len(history))]
predictions = list()
history
126/101:
train = train.tolist()
test = test.tolist()
len(train),len(test)
126/102: series = pd.read_csv('./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
126/103:
X = series.value
train, test = X[1: len(X) - 7], X[len(X) - 7: ]
train, test
126/104:
train = train.tolist()
test = test.tolist()
len(train),len(test)
126/105:
window = 29
model = AutoReg(train, lags=29)
model_fit = model.fit()
coef = model_fit.params
126/106:
history = train[len(train) - window:]
len(history),history
126/107:
history = [history[i] for i in range(len(history))]
predictions = list()
len(history),history
126/108: len(train)
126/109: len(train) - window
126/110: train[len(train) - window:]
126/111:
#isolate the last 29 values of the time series
history = train[len(train) - window:]
126/112: [history[i] for i in range(len(history))]
126/113: len(history)
126/114:
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coeff[0]
    for d in range(window):
        yhat = coeff[d]
126/115:
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0]
    for d in range(window):
        yhat = coef[d+1] * lag[window-d-1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    print(f'Expected: {obs}, Predicted: {yhat}')

rmse = sqrt(mean_squared_error(test,predictions))
print(f'MSE = {rmse}')
128/1:
import os
os.getcwd()
128/2:
import pandas as pd
import matplotlib.pyplot as plt
series = pd.read_csv('./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
128/3: series.head()
128/4:
series['timestamp'] = pd.to_datetime(series['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
plt.plot_date(series['timestamp'],series['value'])
plt.gcf().autofmt_xdate()
128/5: series.set_index('timestamp')
128/6:
values = pd.DataFrame(series.value)
df = pd.concat([values.shift(1),values],axis =1)
df.columns = ['t-1','t']
result = df.corr()
print(result)
128/7:
    autocorr = []
    for lag in range(0,20):
        print("Lag : " + str(lag) + "\n")
        df = pd.concat([values.shift(lag),values],axis =1)
        df.columns = ['t-'+str(lag),'t']
        result = df.corr()
        autocorr.append(result.t[0])
        print(result)
        print("\n\n")
128/8: result.t[0]
128/9: plt.plot([i for i in (autocorr)])
128/10:
from pandas.plotting import lag_plot
series = pd.read_csv('./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
lag_plot(series)
128/11:
from pandas.plotting import autocorrelation_plot
series = pd.read_csv('./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
autocorrelation_plot(series)
128/12:
from statsmodels.graphics.tsaplots import plot_acf
series = pd.read_csv('./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
plot_acf(series['value'], lags=31)
128/13: from sklearn.metrics import mean_squared_error
128/14:
# create lagged dataset
values = pd.DataFrame(series.value)
dataframe = pd.concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
dataframe
128/15:
# split into train and test sets
X = dataframe.values
train, test = X[1:len(X)-7], X[len(X)-7:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
128/16:
# persistence model
def model_persistence(x):
    return x
128/17:
# walk-forward validation
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)
128/18:
# plot predictions vs expected
plt.plot(test_y)
plt.plot(predictions, color='red')
plt.show()
128/19:
from statsmodels.tsa.ar_model import AutoReg
from math import sqrt
128/20: series = pd.read_csv('./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
128/21:
X = series.value
train, test = X[1: len(X) - 7], X[len(X) - 7: ]
128/22:
model = AutoReg(train, lags=29)
model_fit = model.fit()
print(f'Coefficients: {model_fit.params}')
128/23:
predictions = model_fit.predict(start = len(train),end = len(train) + len(test) - 1, dynamic = False)
predictions
128/24:
type(predictions)
len(predictions)
predictions.tolist()
128/25:
#clean the predictions
predictions = predictions.dropna()
predictions
128/26:
for i in range(len(predictions)):
    print(f'predictions: {predictions.tolist()[i]}, Expected: {test.tolist()[i]}')
128/27:
rmse = sqrt(mean_squared_error(test[:len(predictions)],predictions))
print(f"Test RMSE: {rmse}")
128/28:
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
128/29: series = pd.read_csv('./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
128/30:
X = series.value
train, test = X[1: len(X) - 7], X[len(X) - 7: ]
train, test
128/31:
train = train.tolist()
test = test.tolist()
len(train),len(test)
128/32:
window = 29
model = AutoReg(train, lags=29)
model_fit = model.fit()
coef = model_fit.params
128/33:
#isolate the last 29 values of the time series
history = train[len(train) - window:]
128/34:
history = [history[i] for i in range(len(history))]
predictions = list()
len(history),history
128/35:
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0]
    for d in range(window):
        yhat = coef[d+1] * lag[window-d-1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    print(f'Expected: {obs}, Predicted: {yhat}')

rmse = sqrt(mean_squared_error(test,predictions))
print(f'MSE = {rmse}')
128/36: +
128/37:
predictions = model_fit.predict(start = len(train),end = len(train) + len(test) - 1, dynamic = False)
predictions
128/38:
for i in range(len(predictions)):
    print(f'predictions: {predictions.tolist()[i]}, Expected: {test.tolist()[i]}')
128/39:
for i in range(len(predictions)):
    print(f'predictions: {predictions[i]}, Expected: {test.tolist()[i]}')
128/40:
for i in range(len(predictions)):
    print(f'predictions: {predictions[i]}, Expected: {test[i]}')
128/41:
rmse = sqrt(mean_squared_error(test[:len(predictions)],predictions))
print(f"Test RMSE: {rmse}")
128/42:
#isolate the last 29 values of the time series
history = train[len(train) - window:]
128/43:
history = [history[i] for i in range(len(history))]
predictions = list()
len(history),history
128/44:
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0]
    for d in range(window):
        yhat = coef[d+1] * lag[window-d-1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    print(f'Expected: {obs}, Predicted: {yhat}')

rmse = sqrt(mean_squared_error(test,predictions))
print(f'MSE = {rmse}')
128/45:
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0]
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
128/46:
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
128/47: series = pd.read_csv('./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
128/48:
X = series.value
train, test = X[1: len(X) - 7], X[len(X) - 7: ]
train, test
128/49:
train = train.tolist()
test = test.tolist()
len(train),len(test)
128/50:
window = 29
model = AutoReg(train, lags=29)
model_fit = model.fit()
coef = model_fit.params
128/51:
predictions = model_fit.predict(start = len(train),end = len(train) + len(test) - 1, dynamic = False)
predictions
128/52:
for i in range(len(predictions)):
    print(f'predictions: {predictions[i]}, Expected: {test[i]}')
128/53:
rmse = sqrt(mean_squared_error(test[:len(predictions)],predictions))
print(f"Test RMSE: {rmse}")
128/54:
#isolate the last 29 values of the time series
history = train[len(train) - window:]
128/55:
history = [history[i] for i in range(len(history))]
predictions = list()
len(history),history
128/56:
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0]
    for d in range(window):
        yhat = coef[d+1] * lag[window-d-1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    print(f'Expected: {obs}, Predicted: {yhat}')

rmse = sqrt(mean_squared_error(test,predictions))
print(f'MSE = {rmse}')
128/57:
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0]
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
128/58:
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
128/59:
rmse = sqrt(mean_squared_error(test, predictions[len(predictions)/2:]))
print('Test RMSE: %.3f' % rmse)
128/60:
rmse = sqrt(mean_squared_error(test, predictions[len(predictions)//2:]))
print('Test RMSE: %.3f' % rmse)
128/61:
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
128/62:
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
128/63:
# plot
plt.plot(test)
plt.plot(predictions[len(predictions)//2:], color='red')
plt.show()
129/1: import os
129/2: os.getcwd()
129/3: os.chdir(":
129/4: os.chdir("D://Temp/time-series/")
129/5: os.listdir()
129/6: cd data/
129/7: ls
129/8: cd yahoo/
129/9: ls
129/10: cd dataset/
129/11: ls
129/12: cd ydata-labeled-time-series-anomalies-v1_0/
129/13: ls
129/14: cd A1Benchmark/
129/15: runfile('D:/Temp/time-series/untitled1.py', wdir='D:/Temp/time-series')
129/16: runfile('D:/Temp/time-series/untitled1.py', wdir='D:/Temp/time-series')
129/17: runfile('D:/Temp/time-series/untitled1.py', wdir='D:/Temp/time-series')
129/18: os.getcwd()
129/19: cd code
129/20: ls
129/21: runfile("testScriptMutation.py",args=" ../data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_1.csv \"value\" 1")
129/22: M_dataset
129/23: mutated_records
129/24: r0
129/25: len(M_dataset)
129/26: M_dataset.columns.values[0]
129/27: ids
129/28: ids.append(M_dataset.iloc[mutated_records[0]])
129/29: ids
129/30: runfile('D:/Temp/time-series/code/testScriptMutation.py', wdir='D:/Temp/time-series/code')
129/31: runfile("testScriptMutation.py",args=" ../data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_1.csv \"value\" 1")
129/32: ids = []
129/33: ids.append(M_dataset[M_dataset.columns.values[0]].iloc[mutated_records[0]])
129/34: ids
129/35: M_dataset.iloc[[x for x in mutated_records]]
129/36: mutated_records
129/37: runfile('D:/Temp/time-series/code/testScriptMutation.py', wdir='D:/Temp/time-series/code')
129/38: runfile("testScriptMutation.py",args=" ../data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_1.csv \"value\" 1")
129/39: runfile("testScriptMutation.py",args=" ../data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_1.csv \"value\" 1")
130/1: runfile("testScriptMutation.py",args=" ../data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_1.csv \"value\" 2")
130/2: runfile("testScriptMutation.py",args=" ../data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_1.csv \"value\" 2")
130/3: os
130/4: import os
130/5: os.getcwd()
130/6: cd ../data
130/7: ls
130/8: runfile("testScriptMutation.py",args=" ./yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_1.csv \"value\" 2")
130/9: runfile("testScriptMutation.py",args=" ./yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_1.csv \"value\" 3")
130/10: runfile("testScriptMutation.py",args=" ./yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_1.csv \"value\" 4")
130/11: runfile("testScriptMutation.py",args=" ./yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_1.csv \"value\" 5")
131/1: import os
131/2: oc getcwd()
131/3: os.getcwd()
131/4: runfile("testScriptMutation.py",args=" ./yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv \"value\" 1")
131/5: runfile("testScriptMutation.py",args=" ./yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv \"value\" 2")
131/6: runfile("testScriptMutation.py",args=" ./yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv \"value\" 3")
131/7: runfile("testScriptMutation.py",args=" ./yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv \"value\" 4")
131/8: runfile("testScriptMutation.py",args=" ./yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv \"value\" 5")
   1: %history%
   2: !history
   3: %history
   4: %history -g
   5: %history -g > time-series-commands.txt
   6: import os
   7: os.getcwd()
   8: %hist -o -g -f ipython_history.md