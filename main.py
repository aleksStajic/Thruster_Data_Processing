# Eventually reflect reverse direction of thrust vs. current such that thrust is always positive
# Also have a raw thrust vs. current graph potentially

### Import libraries ###
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt; plt.style.use('classic')
import seaborn as sns; sns.set()
from scipy import stats
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
# func: sin_func
# params: time, amplitude and frequency floats
# purpose: returns a typical sign function using the three parameters given
#          not including any phase shift
def sin_func(t, A, f):
    return A*np.sin(2*np.pi*f*t)

### Details for saving figures/files of interest, change naming style as preferred ###
my_path = os.path.dirname(__file__) # path where this script is run from
data_folder = "\\test_data"
ts_file = "\\ts.png"
ct_file_f = "\\ct_f.png"
ct_file_r = "\\ct_r.png"
ts_overlapped_file = "\\ts_overlayed.png"
voltage_fit_file = "\\vfitted_vs_vraw.png"
fig_folder = "\\figures"
show_line = False # change to True if you want to show maximums and minimums on the overlayed chart 

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

### Parse file and collect force, current and voltage data points into lists ###
# Try to make more flexible eventually, as in being able to parse any file loosely formatted the same way 
force = []; current = []; voltage = []; time = []; time2 = []; force_avg = []
force_zero = 0
line = 0
current_line = lines[line].split()
start_test = False
# Once the delay time gets added to the text file I can make the force averaging/background value skipping more efficient
while "<end" not in current_line and line < len(lines):
    if start_test: # Get data points for force, current voltage, ignore values recorded during test delay (5 seconds)
        h,m,s = current_line[1].split(':') # get seconds value corresponding to <timestamp> - 00:00:00.000
        if (float(h) * 3600 + float(m) * 60 + float(s)) - time_zero > test_delay:
            force_zero = np.mean(np.array(force_avg))
            time.append(float(h) * 3600 + float(m) * 60 + float(s)) # time is a list containing floating point second values 
            force.append(float(current_line[2]) - force_zero) # compute all force values wrt the pre-loaded value
            current.append(float(current_line[3]))
            voltage.append(float(current_line[5]))
        else: # During test delay period, get an average for the initial pre-loaded force value
            force_avg.append(float(current_line[2]))
    if lines[line].startswith("#Scaling_Factor"): # Get scaling factors
        sf = lines[line].split()[1:6]
        sf_force = float(sf[0])
        sf_current = float(sf[1])
        sf_voltage = float(sf[3])
        lever_arm_ratio = float(sf[4]) 
    elif lines[line].startswith("#Test_parameters"): # Get sine test parameters
        test_params = current_line
        test_freq = float(test_params[1])
        test_delay = float(test_params[2])
        test_periods = float(test_params[3]) 
        test_phase = float(test_params[4])
        test_amp = float(test_params[5])
    elif "<begin" in current_line: # Start collecting data
        h,m,s = current_line[1].split(':')
        time_zero = float(h) * 3600 + float(m) * 60 + float(s)
        start_test = True
    line += 1
    current_line = lines[line].split()

### Convert force, current and voltage lists into NumPy arrays ###
force = np.array(force) * sf_force * lever_arm_ratio # apply scaling factors 
current = np.array(current) * sf_current
voltage = np.array(voltage) * sf_voltage
power = np.multiply(current, voltage) # compute power
time = np.array(time)

# Format timestamps such that time values are the value in seconds since beginning automated test 
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

### Curve fitting, syncing and period overlappying for time series, using voltage data as the reference "clock" ###
popt, pcov = curve_fit(sin_func, time, voltage, p0=(np.max(voltage), 1/55)) # approximate voltage data to a sine curve
fitted_amp = popt[0]
fitted_freq = popt[1]
fitted_period = 1 / fitted_freq
fitted_voltage = sin_func(time, fitted_amp, fitted_freq) # get approximated voltage sine curve
syncd_data = np.vstack((time, fitted_voltage, force, current)).T # "sync" all plots in a 2D matrix

# Plot fitted voltage curve and raw voltage curve overlayed to assess accuracy of curve fit
fig_fitted, ax_fitted = plt.subplots(1)
fig_fitted.suptitle("Voltage vs time (Curve fit and raw data)", fontsize = 16)
ax_fitted.plot(time, fitted_voltage, '.', label = "Fitted voltage curve")
ax_fitted.plot(time, voltage, '.', label = "Raw data")
ax_fitted.set_ylim(np.amin(voltage) - 20, np.amax(voltage) + 20)
ax_fitted.set_xlabel("Time [s]")
ax_fitted.set_ylabel("Voltage [V]")
plt.legend(loc = "upper right")

# Using the voltage fitted curve, determine the index of the start of each period using a zero cross method
rows = syncd_data.shape[0]
ind_p = [] # list to hold the indices corresponding to the start of each period
ind_p.append(0) # graph starts from 0 since phase shift is 0, so first period starts from index 0
for i in range(rows - 1):
    if(syncd_data[i,1] < 0 and syncd_data[i+1,1] >=0): # if we hit a zero cross going from -ve to +ve, this is the end of one period
        ind_p.append(i+1)
ind_p = np.array(ind_p) # convert to numpy array for consistency
num_periods = len(ind_p)

# Using ind_p, shift all time data in syncd_data such that each period "chunk" starts from 0 
for i in range(1, num_periods - 1): syncd_data[ind_p[i]:ind_p[i+1], 0] -= fitted_period * i
syncd_data[ind_p[num_periods - 1]:len(syncd_data), 0] -= fitted_period * (num_periods - 1) # time shift last period

# Sort sync_d by time and store in a new DataFrame -> This DataFrame holds our synced, overlapped data
sorted_data = syncd_data[np.argsort(syncd_data[:, 0])] # argsort returns the indices that would sort the array specified
df_sorted = pd.DataFrame(sorted_data, columns = ["time", "voltage", "force", "current"])

# Extract values in DataFrame into individual Numpy arrays for convenience -> these arrays hold the "sorted" data points over 5 periods
v_avg = np.array(df_sorted["voltage"]) # voltage [V]
i_avg = np.array(df_sorted["current"]) # current [A]
T_avg = np.array(df_sorted["force"]) # thrust [kgf]
t = np.array(df_sorted["time"]) # time [s]
p_avg = np.multiply(v_avg, i_avg) # power [W]

#plot all the synced and overlapped data
fig_overlapped, ax_sync = plt.subplots(1)
fig_overlapped.suptitle("Current, force vs. time synced to LSE voltage vs. time fit with periods overlapped", fontsize = 16)
ax_sync.plot(t, v_avg, '.', label = "Voltage [V]") # voltage
ax_sync.plot(t, i_avg, '.', label = "Current [A]") # current
ax_sync.plot(t, T_avg, '.', label = "Thrust [kgf]") # force
ax_sync.set_xlabel("Time [s]")

### Determine and store peak values from synced and overlapped data ###
# Determine max power for reverse direction
pmin = p_avg[0]
for i in range(len(power)): 
    if(abs(p_avg[i]) > abs(pmin) and v_avg[i] < 0 and i_avg[i] < 0): pmin = p_avg[i]

# Store peak values from synced and overlapped data into Pandas dataframe 
peaks = np.array([np.amax(T_avg), np.amax(i_avg), np.amax(v_avg), np.amax(p_avg)])
troughs = np.array([abs(np.amin(T_avg)), abs(np.amin(i_avg)), abs(np.amin(v_avg)), pmin])
df_extrema = pd.DataFrame({"Max forward": peaks, "Max reverse": troughs}, index = ["Force [lbf]", "Current [A]", "Voltage [V]", "Power [W]"])

# Graph peak values on plot of overlapped data
if show_line:
    v_max = np.full_like(np.arange(int(len(v_avg)), dtype = float), peaks[2])
    v_min = np.full_like(np.arange(int(len(v_avg)), dtype = float), troughs[2])
    i_max = np.full_like(np.arange(int(len(i_avg)), dtype = float), peaks[1])
    i_min = np.full_like(np.arange(int(len(i_avg)), dtype = float), troughs[1])
    T_max = np.full_like(np.arange(int(len(T_avg)), dtype = float), peaks[0])
    T_min = np.full_like(np.arange(int(len(T_avg)), dtype = float), troughs[0])

    ax_sync.plot(t[0:int(len(t) / 2)], v_max[0:int(len(t) / 2)], '.', label = "Voltage maximum"); ax_sync.plot(t[int(len(t) / 2):int(len(t))], -v_min[int(len(t) / 2):int(len(t))], '.', label = "Voltage minimum")
    ax_sync.plot(t[0:int(len(t) / 2)], i_max[0:int(len(t) / 2)], '.', label = "Current maximum"); ax_sync.plot(t[int(len(t) / 2):int(len(t))], -i_min[int(len(t) / 2):int(len(t))], '.', label = "Current minimum")
    ax_sync.plot(t[0:int(len(t) / 2)], T_max[0:int(len(t) / 2)], '.', label = "Thrust maximum"); ax_sync.plot(t[int(len(t) / 2):int(len(t))], -T_min[int(len(t) / 2):int(len(t))], '.', label = "Thrust minimum")

    # Plot vertical line at the time intersection of max values for voltage and current
    ax_sync.plot(np.full_like(np.arange(int(len(v_avg)), dtype = float), t[np.argmax(v_avg)]), v_avg, '.')
    ax_sync.plot(np.full_like(np.arange(int(len(i_avg)), dtype = float), t[np.argmax(i_avg)]), i_avg, '.')
plt.legend(loc = "upper right")

### Graph thrust vs. current ###
# Seperate forward/reverse direction current and thrust values, using fitted voltage plot as zero cross reference
i = 0
current_f = []; thrust_f = []; 
current_r = []; thrust_r = []; 

# Use fitted voltage curve and zero cross method to determine forward/reverse values from the overlapped, synced data
while(v_avg[i] < 0): i += 1 # get to zero cross of voltage graph
while(v_avg[i] >= 0): # forward 
    current_f.append(i_avg[i])
    thrust_f.append(T_avg[i])
    i += 1
while(i < len(voltage) - 1): # reverse
    current_r.append(i_avg[i])
    thrust_r.append(T_avg[i])
    i += 1

# Convert lists to NumPy arrays
current_f = np.array(current_f); thrust_f = np.array(thrust_f)
current_r = np.array(current_r); thrust_r = np.array(thrust_r)

### Use Linear Regression to determine thrust vs. current slopes (forward and reverse) ###
# Determine forward direction thrust vs. current trendline
slope, intercept, r, p, std_err = stats.linregress(current_f, thrust_f)
m_f = slope; b_f = intercept; r_f = r

def ct_line(current):
    return slope * current + intercept

mymodel_f = list(map(ct_line, current_f))

# Determine reverse direction thrust vs. current trendline
slope, intercept, r, p, std_err = stats.linregress(current_r, thrust_r)
m_r = slope; b_r = intercept; r_r = r

mymodel_r = list(map(ct_line, current_r))

### Determine length of dead band for thrust vs. current ###
deadband = abs(b_f - b_r)

### Save slope, intercept and r^2 data into dataframe ###
ct_stats_f = np.array([m_f, b_f, r_f**2])
ct_stats_r = np.array([m_r, b_r, r_r**2])
df_stats = pd.DataFrame({"Forward": ct_stats_f, "Reverse": ct_stats_r}, index = ["m [lbf/A]", "b", "r^2"])

# Graph forward direction thrust vs. current
figc_f = plt.figure()
figc_f.suptitle("Thrust vs. current (forward)", fontsize = 16)
axc_f = plt.axes()

axc_f.set_ylabel("Thrust [kgf]")
axc_f.set_xlabel("Current [A]")

plt.scatter(current_f, thrust_f)
plt.plot(current_f, mymodel_f, color = "orange") 

# Graph reverse direction thrust vs. current
figc_r = plt.figure()
figc_r.suptitle("Thrust vs. current (reverse)", fontsize = 16)
axc_r = plt.axes()

axc_r.set_ylabel("Thrust [kgf]")
axc_r.set_xlabel("Current [A]")

plt.scatter(current_r, thrust_r)
plt.plot(current_r, mymodel_r, color = "orange")

### Save figures and files ###
if not os.path.isdir(my_path + data_folder + fig_folder):
    os.makedirs(my_path + data_folder + fig_folder)
figt.savefig(my_path + data_folder + fig_folder + ts_file)
figc_f.savefig(my_path + data_folder + fig_folder + ct_file_f)
figc_r.savefig(my_path + data_folder + fig_folder + ct_file_r)
fig_overlapped.savefig(my_path + data_folder + fig_folder + ts_overlapped_file)
fig_fitted.savefig(my_path + data_folder + fig_folder + voltage_fit_file)

with open(my_path + data_folder + "\\stats.txt", mode = 'w') as file_object:
    print("|Peak Values|", file = file_object)
    print(df_extrema, file = file_object)
    print("\n|Thrust vs. current stats|", file = file_object)
    print(df_stats, file = file_object)
    print("\nThrust vs. current deadband magnitude: ", deadband, file = file_object)

### Display all plots, data of interest -> last action in program, since plt.show() stalls execution ###
#plt.show() 

