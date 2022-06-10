# Import libraries #
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt; plt.style.use('classic')
import seaborn as sns; sns.set()

# Function definitions #
# func: get_seconds
# param: formatted timestamp string following hh:mm:ss.000
# returns: seconds value corresponding to <timestamp> - 00:00:00.000
def get_seconds(time_str):
    h,m,s = time_str.split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)

# Details for saving figures of interest, change as preferred #
ts_file = r"\ts.png"
ct_file = r"\ct.png"
fig_folder = r"\Figures"
my_path = r"C:\Users\seamo\OneDrive\Desktop\learnpython\Thruster_Data_Processing"

# Read .log file #
file_location = r"C:\Users\seamo\OneDrive\Desktop\Thruster_Tests\20220610_S_100_50_5_STD_STD_NoFilters_GND1ShortedBoth.log" 
try:
    fin = open(file_location, 'r') # fin is assigned to the file object returned by open()
    lines = fin.readlines() # readlines() method returns a list containing each line of the file in string format
    fin.close() # close file
except IOError:
    print("Error, file does not exist\nTerminating program...")
    quit() # terminate program if file cannot be opened

# At this point, lines[] contains each line of the text file in string format

# Parse file and collect force, current and voltage data points into lists #
# Try to make modular eventually, as in being able to parse any file loosely formatted the same way 
force = []; current = []; voltage = []; time = []
for line in range(len(lines) - 1):
    if lines[line].startswith("#Scaling_Factor"): # Get scaling factors
        sf = lines[line].split()[1:6]
        sf_force = float(sf[0])
        sf_current = float(sf[1])
        sf_voltage = float(sf[3])
        lever_arm_ratio = float(sf[4]) 
    elif line >= 5: # Get data points for force, current voltage
        current_line = lines[line].split()
        time.append(get_seconds(current_line[1])) # time is a list containing floating point second values 
        force.append(float(current_line[2]))
        current.append(float(current_line[3]))
        voltage.append(float(current_line[5]))

# At this point, force[], current[] and voltage[] contain all respective raw data points in list format

# Convert force, current and voltage lists into NumPy arrays #
force = np.array(force) * sf_force # apply scaling factors 
current = np.array(current) * sf_current
voltage = np.array(voltage) * sf_voltage
time = np.array(time)

# Compute power #
power = np.multiply(current, voltage)

# Format timestamps such that time values are the value in seconds since beginning automated test #
ref_time = time[0]
for t in range(len(time)):
    time[t] = time[t] - ref_time

# Create dataframe with Pandas # 
df_data = pd.DataFrame({"Force": force, "Current": current, "Voltage": voltage, "Power": power}, index = time)

# Graph time series plots with Matplotlib #
figt, axt = plt.subplots(3, constrained_layout = True) #creates a figure and a set of subplots, subplots() returns tuple of (Figure, Axes)
figt.suptitle("Current, voltage, force vs. time (raw)", fontsize = 16)
for plot in range(len(axt)): axt[plot].set_xlabel("Time [s]")

axt[0].plot(df_data.index, df_data["Current"], '.') # Current vs. time scatter
axt[0].set_ylabel("Current (raw) [A]")
axt[0].set_ylim(-15,15)

axt[1].plot(df_data.index, df_data["Voltage"], '.') # Voltage vs. time scatter
axt[1].set_ylabel("Voltage (raw) [A]")

axt[2].plot(df_data.index, df_data["Force"], '.') # Force/thrust vs. time scatter
axt[2].set_ylabel("Force (raw) [kgf]")

# Display peak values from raw data #
peaks = np.array([np.amax(force), np.amax(current), np.amax(voltage), np.amax(power)])
troughs = np.array([np.min(force), np.amin(current), np.amin(voltage), np.amin(power)])
df_peaks = pd.DataFrame({"Max (raw)": peaks, "Min (raw)": troughs}, index = ["Force", "Current", "Voltage", "Power"])
print("\n", df_peaks)

# Graph thrust vs. current using Matplotlib #
figc = plt.figure()
figc.suptitle("Thrust vs. current", fontsize = 16)
axc = plt.axes()
axc.plot(df_data["Current"], df_data["Force"], '.')
axc.set_ylabel("Thrust [kgf]")
axc.set_xlabel("Current [A]")

# Save figures #
if not os.path.isdir(my_path + fig_folder):
    os.makedirs(my_path + fig_folder)
figt.savefig(fig_folder[1:len(fig_folder)] + ts_file)
figc.savefig(fig_folder[1:len(fig_folder)] + ct_file)

# Pre-process data (clean up rails/outliers, re-sample using pandas) #

# Display all plots -> last action in program #
#plt.show() 
# For power, I need to figure out how to plot max power for reverse direction