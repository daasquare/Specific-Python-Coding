import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats

# Read the Excel workbook
workbook_path = '/Users/akin.o.akinjogbin/Desktop/July 2023 Tine Testing Raw data/Tine Stiffness/Tine Stiffness/ProcessedData.xlsx'
output_path = os.path.dirname(workbook_path)  # Get the directory of the processed data file
xls = pd.ExcelFile(workbook_path)

# Create a list to store the results for each sheet
results = []

# Create a single figure with a grid layout for the subplots
num_sheets = len(xls.sheet_names)
num_cols = 3  # You can adjust the number of columns as per your preference
num_rows = (num_sheets + num_cols - 1) // num_cols

# Determine the number of subplots and create the figure and axes
fig, axs = plt.subplots(num_rows, num_cols, figsize=(25, 5*num_rows), sharex=True)

# Flatten the axs array for easier iteration
axs = axs.flatten()

# Iterate over each sheet and plot in the respective subplot
for i, sheet_name in enumerate(xls.sheet_names):
    ax = axs[i]  # Get the current subplot for the current sheet

    # Read the sheet into a DataFrame
    df = pd.read_excel(workbook_path, sheet_name=sheet_name)

    # Extract data from specific columns using header names
    x = df['Distance (mm)'].values
    y = df['Raw Load (N)'].values

    # Apply Butterworth filter
    order = 1  # Filter order
    cutoff_freq = 0.05  # Cutoff frequency (adjust as needed)
    b, a = signal.butter(order, cutoff_freq, fs=1)  # Apply Butterworth filter
    filtered_y = signal.filtfilt(b, a, y)

    # Calculate absolute value of filtered y
    abs_filtered_y = np.abs(filtered_y)

    # Find the indices of local maxima (peaks)
    peak_indices = signal.argrelextrema(abs_filtered_y, np.greater)[0]

    # Skip initial peaks before X=0.5 and plot the remaining data
    for peak_index in peak_indices:
        if x[peak_index] <= 0.5:
            continue

        # Plot the absolute filtered data
        ax.plot(x, abs_filtered_y, label='Absolute Filtered Data')

        # Plot the initial peak
        ax.plot(x[peak_index], abs_filtered_y[peak_index], 'ro', label='Initial Peak')

        # Fit linear regression from (X=0, Y=0) to the identified initial peak
        tangent_slope = abs_filtered_y[peak_index] / x[peak_index]  # Slope for tangent line
        tangent_intercept = 0  # Intercept for tangent line (passes through Y=0)

        # Calculate the regression line
        regression_line = tangent_slope * x + tangent_intercept

        # Plot the regression line for the current sheet only
        ax.plot(x, regression_line, 'g--', label='Tangent Line' if i == 0 else '', alpha=0.7)

        # Set labels and title for each subplot
        ax.set_xlabel('Distance (mm)')
        ax.set_ylabel('Force (N)')
        ax.set_title('Tine Stiffness Plot - {}'.format(sheet_name))

        # Add legend to each subplot
        ax.legend()

        # Append the result to the list for each sheet
        results.append({"Sheet Name": sheet_name, "Gradient": tangent_slope})
        break  # Skip to the next sheet after processing the first peak

# Hide any remaining empty subplots
for i in range(num_sheets, num_rows*num_cols):
    axs[i].axis('off')

# Adjust layout and add a common title for the entire figure
plt.tight_layout()  # To prevent overlapping of subplots
plt.suptitle('Tine Stiffness Plots for all Sheets')

# Show the combined plots
plt.show()

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)

# Save the DataFrame to an xlsx file in the same directory as the processed data
output_filename = os.path.join(output_path, 'LinearRegressionResults.xlsx')
results_df.to_excel(output_filename, index=False)

print("Results saved to:", output_filename)
