# import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# define plot properties
SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

# load data
df = pd.read_csv("results.csv")
df["Size"] = df.Size.astype('float')

# plot data
fig, ax = plt.subplots()
plt.scatter(df["Size"], df["Hit @ 1"], label = "Hit @ 1")
plt.scatter(df["Size"], df["Hit @ 2"], label = "Hit @ 2")
plt.scatter(df["Size"], df["Hit @ 3"], label = "Hit @ 3")
plt.ylim((0, 85))
plt.ylim((0, 85))
plt.ylim((0, 85))
plt.xlim((0, None))
plt.xlim((0, None))
plt.xlim((0, None))

# prepare data for annotation and log line of best fit
X = df["Size"].tolist()
Y = df["Hit @ 1"].tolist()
Y2 = df["Hit @ 2"].tolist()
Y3 = df["Hit @ 3"].tolist()
debertalessX = X[:6] + X[7:13]
debertalessY = Y[:6] + Y[7:13]
debertalessY2 = Y2[:6] + Y2[7:13]
debertalessY3 = Y3[:6] + Y3[7:13]
names = df["Model"].tolist()

# annotate the deberta entries
plt.annotate(names[6] +', '+names[13], (X[6], Y[6]))
plt.annotate(names[6], (X[6], Y2[6]))
plt.annotate(names[6], (X[6], Y3[6]))
plt.annotate(names[13], (X[13], Y3[13]))
plt.annotate(names[13], (X[13], Y2[13]))

# plot log lines of best fit ignoring deberta
slope1, intercept1 = np.polyfit(np.log(debertalessX), debertalessY, 1)
plt.plot(np.arange(1, 8e8, 1000), slope1 * np.log(np.arange(1, 8e8, 1000)) + intercept1, '--')
slope2, intercept2 = np.polyfit(np.log(debertalessX), debertalessY2, 1)
plt.plot(np.arange(1, 8e8, 1000), slope2 * np.log(np.arange(1, 8e8, 1000)) + intercept2, '--')
slope3, intercept3 = np.polyfit(np.log(debertalessX), debertalessY3, 1)
plt.plot(np.arange(1, 8e8, 1000), slope3 * np.log(np.arange(1, 8e8, 1000)) + intercept3, '--')

# formatting
plt.title(f'Model Accuracies Plotted Against Model Size')
plt.ylabel(f'Model Accuracies')
plt.xlabel("Model Number of Parameters")
leg = plt.legend()
leg.set_draggable(state=True)
plt.grid()
plt.show()