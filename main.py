### Do another flow chart mapping out algorithm -> don't get lost in weeds
### Use the voltage fitted plot to find peak values instead of looking at peak values in arrays
### can then use time indexing with the approximated period to find peak values for each plot
### maybe just curve fit all plots to find peak values??? need to check if the values are within error margins

### Import libraries ###
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt; plt.style.use('classic')
import seaborn as sns; sns.set()
from scipy import stats
import scipy.optimize
from scipy.optimize import curve_fit

### Function definitions ###
# func: user_in
# param: string that prompts the user for input
# purpose: prompts the user to enter data through the terminal
#          attempts to handle invalid user input until a valid input is entered 
# returns: the value entered by the user (could be str, float, int, etc.)
def user_in(msg):
    while True:
        try:
            result = input(msg)
        except:
            continue
        else:
            return result

### Details for saving figures/files of interest, change naming style as preferred ###
my_path = os.path.dirname(__file__) # path where this script is run from
data_folder = "\\test_data"
ts_file = "\\ts.png"
ct_file_f = "\\ct_f.png"
ct_file_r = "\\ct_r.png"
fig_folder = "\\figures"

### Read .log file, handling invalid user input ###
data_file = user_in("Enter data file path: ")
while True:
    try:
        fin = open(data_file, 'r') # fin is assigned to the file object returned by open()
    except IOError:
        print("Error, file entered does not exist, please re-type")
        data_file = user_in("Enter data file path: ")
    else:
        lines = fin.readlines() # readlines() method returns a list containing each line of the file in string format
        fin.close() 
        break

# At this point, lines[] contains each line of the text file in string format as a list

### Parse file and collect force, current and voltage data points into lists ###
# Try to make more flexible eventually, as in being able to parse any file loosely formatted the same way 
force = []; current = []; voltage = []; time = []; time2 = []; force_avg = []
force_zero = 0
line = 0
current_line = lines[line].split()
while "<end" not in current_line and line < len(lines):
    if lines[line].startswith("#Scaling_Factor"): # Get scaling factors
        sf = lines[line].split()[1:6]
        sf_force = float(sf[0])
        sf_current = float(sf[1])
        sf_voltage = float(sf[3])
        lever_arm_ratio = float(sf[4]) 
    elif line >= 18: # Get data points for force, current voltage, ignore values recorded during test delay (5 seconds)
        h,m,s = current_line[1].split(':') # get seconds value corresponding to <timestamp> - 00:00:00.000
        time.append(float(h) * 3600 + float(m) * 60 + float(s)) # time is a list containing floating point second values 
        force.append(float(current_line[2]) - force_zero) # compute all force values wrt the pre-loaded value
        current.append(float(current_line[3]))
        voltage.append(float(current_line[5]))
    elif 5 < line < 18: # During test delay period, get an average for the initial pre-loaded force value
        force_avg.append(float(current_line[2]))
        force_zero = np.mean(np.array(force_avg))
    line += 1
    current_line = lines[line].split()

# At this point, force[], current[] and voltage[] contain all respective raw data points in list format

### Convert force, current and voltage lists into NumPy arrays ###
force = np.array(force) * sf_force * lever_arm_ratio # apply scaling factors 
current = np.array(current) * sf_current
voltage = np.array(voltage) * sf_voltage
power = np.multiply(current, voltage) # compute power
time = np.array(time)

# Format timestamps such that time values are the value in seconds since beginning automated test #
ref_time = time[0]
for t in range(len(time)): time[t] = time[t] - ref_time

### Create dataframe with Pandas, indexing by time in seconds ###
df_data = pd.DataFrame({"Force": force, "Current": current, "Voltage": voltage, "Power": power}, index = time)

### Graph raw time series plots with Matplotlib ###
figt, axt = plt.subplots(3, constrained_layout = True) #creates a figure and a set of subplots, subplots() returns tuple of (Figure, Axes)
figt.suptitle("Current, voltage, force vs. time (raw)", fontsize = 16)
for plot in range(len(axt)): axt[plot].set_xlabel("Time [s]")

axt[0].plot(df_data.index, df_data["Current"], '.') # Current vs. time scatter
axt[0].set_ylabel("Current (raw) [A]")

axt[1].plot(df_data.index, df_data["Voltage"], '.') # Voltage vs. time scatter
axt[1].set_ylabel("Voltage (raw) [A]")

axt[2].plot(df_data.index, df_data["Force"], '.') # Force/thrust vs. time scatter
axt[2].set_ylabel("Force (raw) [kgf]")

# Determine max power for reverse direction
pmin = power[0]
for i in range(len(power)): 
    if(abs(power[i]) > abs(pmin) and voltage[i] < 0 and current[i] < 0): pmin = power[i]

### Determine and store peak values from raw data into Pandas dataframe ###
peaks = np.array([np.amax(force), np.amax(current), np.amax(voltage), np.amax(power)])
troughs = np.array([abs(np.min(force)), abs(np.amin(current)), abs(np.amin(voltage)), pmin])
df_extrema = pd.DataFrame({"Max forward (raw)": peaks, "Max reverse (raw)": troughs}, index = ["Force [lbf]", "Current [A]", "Voltage [V]", "Power [W]"])
# add units

### Curve fitting for time series ###
def sin_func(t, A, f):
    return A*np.sin(2*np.pi*f*t)
popt, pcov = curve_fit(sin_func,  # function to approximate data
                       time,      # measured x values
                       voltage,   # measured y values
                       p0=(np.max(voltage), 1.0/55.0))  # the initial guess for amplitude and frequency
fitted_amp = popt[0]
fitted_freq = popt[1]
syncd_v_data = sin_func(time, fitted_amp, fitted_freq) # approximated voltage sine curve
syncd_data = np.vstack((time, syncd_v_data, force, current)).T # "syncing" all plots in a array stack

# Obtain max, min and zero values for fitted voltage plot
v_max = np.full_like(syncd_data[:,1],syncd_data[:,1].max()) # max for voltage plot
v_min = np.full_like(syncd_data[:,1],syncd_data[:,1].min()) # min for voltage plot
v_zero = np.full_like(syncd_data[:,1],0) # zero for plot

#plot all the synced data 
fig, figsync = plt.subplots(1)
figsync.plot(syncd_data[:,0],syncd_data[:,1]) 
figsync.plot(syncd_data[:,0],syncd_data[:,3])
figsync.plot(syncd_data[:,0],syncd_data[:,2])

### Graphing thrust vs. current ###
# Seperate forward/reverse direction current, thrust and voltage values, using voltage plot as zero cross reference
num_periods = 5
compared = False
voltage_avg_f = np.array([])
i = j = 0
current_f = []; force_f = []; voltage_f = []
current_r = []; force_r = []; voltage_r = []

# Redo such that loops are less dependent on sign -> use time values and properties of sine wave
# Makes the assumption that after crossing 0, voltage plot does not dip negative
while(voltage[i] <= 0 and i < len(voltage) - 1): i += 1 # get to zero cross of voltage graph
for j in range(num_periods):
    while(voltage[i] >= 0 and i < len(voltage) - 1): # get forward current, force and voltage values
        current_f.append(current[i])
        force_f.append(force[i])
        voltage_f.append(voltage[i])
        i += 1
    while(voltage[i] < 0 and i < len(voltage) - 1): # get reverse current, force and voltage values
        current_r.append(current[i])
        force_r.append(force[i])
        voltage_r.append(voltage[i])
        i += 1

# Convert lists to NumPy arrays
current_f = np.array(current_f); force_f = np.array(force_f); voltage_f = np.array(voltage_f)
current_r = np.array(current_r); force_r = np.array(force_r); voltage_r = np.array(voltage_r)

### Use Linear Regression to determine thrust vs. current slopes (forward and reverse) ###
# Determine forward direction thrust vs. current trendline
slope, intercept, r, p, std_err = stats.linregress(current_f, force_f)
m_f = slope; b_f = intercept; r_f = r

def ct_line(current):
  return slope * current + intercept

mymodel_f = list(map(ct_line, current_f))

# Determine reverse direction thrust vs. current trendline
slope, intercept, r, p, std_err = stats.linregress(current_r, force_r)
m_r = slope; b_r = intercept; r_r = r

mymodel_r = list(map(ct_line, current_r))

### Save slope and intercept data into dataframe ###
ct_stats_f = np.array([m_f, b_f, r_f**2])
ct_stats_r = np.array([m_r, b_r, r_r**2])
df_stats = pd.DataFrame({"Forward": ct_stats_f, "Reverse": ct_stats_r}, index = ["m [lbf/A]", "b", "r^2"])

# Graph forward direction thrust vs. current
figc_f = plt.figure()
figc_f.suptitle("Thrust vs. current (forward)", fontsize = 16)
axc_f = plt.axes()

axc_f.set_ylabel("Thrust [kgf]")
axc_f.set_xlabel("Current [A]")

plt.scatter(current_f, force_f)
plt.plot(current_f, mymodel_f) 

# Graph reverse direction thrust vs. current
figc_r = plt.figure()
figc_r.suptitle("Thrust vs. current (reverse)", fontsize = 16)
axc_r = plt.axes()

axc_r.set_ylabel("Thrust [kgf]")
axc_r.set_xlabel("Current [A]")

plt.scatter(current_r, force_r)
plt.plot(current_r, mymodel_r)

### Determine thrust vs. current intercepts -> get length of dead band ###

### Save figures and files ###
if not os.path.isdir(my_path + data_folder + fig_folder):
    os.makedirs(my_path + data_folder + fig_folder)
figt.savefig(my_path + data_folder + fig_folder + ts_file)
figc_f.savefig(my_path + data_folder + fig_folder + ct_file_f)
figc_r.savefig(my_path + data_folder + fig_folder + ct_file_r)

with open(my_path + data_folder + "\\stats.txt", mode = 'w') as file_object:
    print("|Peak Values|", file = file_object)
    print(df_extrema, file = file_object)
    print("\n|Thrust vs. current stats|", file = file_object)
    print(df_stats, file = file_object)

### Display all plots, data of interest -> last action in program ###
#plt.show() 


