### Import libraries ###
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt; plt.style.use('classic')
import seaborn as sns; sns.set()

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
my_path = os.path.dirname(__file__) 
data_folder = "\\test_data"
ts_file = "\\ts.png"
ct_file = "\\ct.png"
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
# Try to make modular eventually, as in being able to parse any file loosely formatted the same way 
force = []; current = []; voltage = []; time = []
line = 0
current_line = lines[line].split()
while "<end" not in current_line and line < len(lines) - 1:
    if lines[line].startswith("#Scaling_Factor"): # Get scaling factors
        sf = lines[line].split()[1:6]
        sf_force = float(sf[0])
        sf_current = float(sf[1])
        sf_voltage = float(sf[3])
        lever_arm_ratio = float(sf[4]) 
    elif line >= 5: # Get data points for force, current voltage
        h,m,s = current_line[1].split(':') # get seconds value corresponding to <timestamp> - 00:00:00.000
        time.append(float(h) * 3600 + float(m) * 60 + float(s)) # time is a list containing floating point second values 
        force.append(float(current_line[2]))
        current.append(float(current_line[3]))
        voltage.append(float(current_line[5]))
    line += 1
    current_line = lines[line].split()

# At this point, force[], current[] and voltage[] contain all respective raw data points in list format

### Convert force, current and voltage lists into NumPy arrays ###
force = np.array(force) * sf_force # apply scaling factors 
current = np.array(current) * sf_current
voltage = np.array(voltage) * sf_voltage
time = np.array(time)

### Compute power ###
power = np.multiply(current, voltage)

# Format timestamps such that time values are the value in seconds since beginning automated test #
ref_time = time[0]
for t in range(len(time)):
    time[t] = time[t] - ref_time

### Create dataframe with Pandas ###
df_data = pd.DataFrame({"Force": force, "Current": current, "Voltage": voltage, "Power": power}, index = time)

### Graph time series plots with Matplotlib ###
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

### Graph thrust vs. current using Matplotlib ###
figc = plt.figure()
figc.suptitle("Thrust vs. current", fontsize = 16)
axc = plt.axes()

axc.plot(df_data["Current"], df_data["Force"], '.')
axc.set_ylabel("Thrust [kgf]")
axc.set_xlabel("Current [A]")

# For power, I need to figure out how to plot max power for reverse direction (this will do for now)
pmin = power[0]
for i in range(len(power) - 1):
    if(abs(power[i]) > abs(pmin) and voltage[i] < 0 and current[i] < 0): pmin = power[i]

### Determine and store peak values from raw data into Pandas dataframe ###
peaks = np.array([np.amax(force), np.amax(current), np.amax(voltage), np.amax(power)])
troughs = np.array([np.min(force), np.amin(current), np.amin(voltage), pmin])
df_extrema = pd.DataFrame({"Max forward (raw)": peaks, "Max reverse (raw)": troughs}, index = ["Force", "Current", "Voltage", "Power"])

### Save figures and files ###
if not os.path.isdir(my_path + data_folder + fig_folder):
    os.makedirs(my_path + data_folder + fig_folder)
figt.savefig(my_path + data_folder + fig_folder + ts_file)
figc.savefig(my_path + data_folder + fig_folder + ct_file)

with open(my_path + data_folder + "\\extrema.txt", mode = 'w') as file_object:
    print(df_extrema, file = file_object)

### Pre-process data (clean up rails/outliers, re-sample using pandas) ###

### Display all plots, data of interest -> last action in program ###
print("\n", df_extrema)
plt.show() 


