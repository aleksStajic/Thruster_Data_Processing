# Import libraries #
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Read .log file #
file_location = r"C:\Users\seamo\OneDrive\Desktop\Thruster_Tests\20220603_S_100_50_5_STD_STD_Filter_Output.log"
# r flag indicates a "raw" string, which means backslashes are treated as characters, not escape sequences 
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
    elif line >= 5: # Get data points, applying scaling factors 
        current_line = lines[line].split()
        time.append(current_line[1])
        force.append(float(current_line[2]) * sf_force)
        current.append(float(current_line[3]) * sf_current)
        voltage.append(float(current_line[5]) * sf_voltage)

# At this point, force[], current[] and voltage[] contain all respective raw data points in list format

# Convert force, current and voltage lists into NumPy arrays #
force = np.array(force)
current = np.array(current)
voltage = np.array(voltage)

# Create dataframe with Pandas # 
df = pd.DataFrame({"Force": force, "Current": current, "Voltage": voltage}, index = time)
print(df.Force)