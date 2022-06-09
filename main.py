# Import libraries #
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt ; plt.style.use('classic')
import seaborn as sns; sns.set()

# Function definitions #
# func: get_seconds
# param: formatted timestamp string following hh:mm:ss.000
# returns: seconds value corresponding to <timestamp> - 00:00:00.000
def get_seconds(time_str):
    h,m,s = time_str.split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)

# Read .log file #
file_location = r"C:\Users\seamo\OneDrive\Desktop\Thruster_Tests\20220603_S_100_50_5_STD_STD_Filter_Output.log" 
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
force = np.array(force) * sf_force
current = np.array(current) * sf_current
voltage = np.array(voltage) * sf_voltage
time = np.array(time)

# Format timestamps such that time values are the value in seconds since beginning automated test #
ref_time = time[0]
for t in range(len(time)):
    time[t] = time[t] - ref_time

# Create dataframe with Pandas # 
df_data = pd.DataFrame({"Force": force, "Current": current, "Voltage": voltage}, index = time)

# Graph time series plots with Matplotlib #
fig, ax = plt.subplots(3) 
for plot in range(len(ax)): ax[plot].set_xlabel("Time [s]")

ax[0].plot(df_data.index, df_data.Current, '.') # Current vs. time
ax[0].set_ylabel("Current (raw) [A]")
ax[0].set_ylim(-10,15)

ax[1].plot(df_data.index, df_data.Voltage, '.') # Voltage vs. time
ax[1].set_ylabel("Voltage (raw) [A]")

ax[2].plot(df_data.index, df_data.Force, '.') # Force/thrust vs. time
ax[2].set_ylabel("Force (raw) [kgf]")

plt.show()

# Display peak values from raw data #
peaks = np.array([force.max(), current.max(), voltage.max()])
troughs = np.array([force.min(), current.min(), voltage.min()])
df_peaks = pd.DataFrame({"Max (raw)": peaks, "Min (raw)": troughs}, index = ["Force", "Current", "Voltage"])
print(df_peaks.index)