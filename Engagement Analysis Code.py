import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from matplotlib.widgets import CheckButtons

# Read the Excel file
excel_file = pd.ExcelFile('/Users/akin.o.akinjogbin/Desktop/July 2023 Tine Testing Raw data/Tine Engagement /Proccessed Files/Versions Runs/ProcessedData.xlsx')

# Create a figure and a single subplot
fig, ax = plt.subplots(figsize=(20, 20))

# Create an empty DataFrame to store the peak coordinates
peak_data = pd.DataFrame(columns=['Sheet', 'Peak Position(mm)', 'Peak Force(N)', 'Steady State Position(mm)', 'Steady State Force(N)', 'Window Number'])

# Create a list to store the visibility status of each line
line_visibility = []

# Function to create virtual windows after X=20, get the index of the Max Peak within each window,
# and return the X and Y values at the end of each window
def get_max_peak_in_windows(x_data, y_data, window_size, num_windows, start_x):
    max_peak_idx = []
    end_x_values = []
    end_y_values = []
    total_data_points = len(x_data)

    # Find the index of the first element in x_data that is greater than or equal to start_x
    start_idx = np.argmax(x_data >= start_x)

    step = (total_data_points - start_idx) // num_windows

    for i in range(num_windows):
        window_start_idx = start_idx + i * step
        window_end_idx = min(start_idx + (i + 1) * step, total_data_points)
        window_x = x_data[window_start_idx:window_end_idx]
        window_y = y_data[window_start_idx:window_end_idx]
        peak_idx, _ = find_peaks(window_y)

        if len(peak_idx) > 0:
            max_peak_idx.append(window_start_idx + peak_idx[np.argmax(window_y[peak_idx])])

        # Store the X and Y values at the end of the window
        end_x_values.append(x_data[window_end_idx - 1])
        end_y_values.append(y_data[window_end_idx - 1])

        # Plot the virtual windows in black
        ax.axvspan(window_x[0], window_x[-1], alpha=0.2, color='black')

    return max_peak_idx, end_x_values, end_y_values


# Iterate over each sheet in the workbook
for i, sheet_name in enumerate(excel_file.sheet_names):
    # Read the sheet data
    data = pd.read_excel(excel_file, sheet_name=sheet_name)

    # Specify column numbers for x and y
    x_column = 2  # Column number for x (starting from 0)
    y_column = 0  # Column number for y (starting from 0)

    # Extract the x and y columns from the sheet data
    x = data.iloc[:, x_column].values
    y = data.iloc[:, y_column].values

    # Convert y values to absolute values
    y = np.abs(y)

    # Set the filter order and cutoff frequency
    filter_order = 5  # Adjust the filter order as needed
    cutoff_freq = 0.05  # Adjust the cutoff frequency as needed

    # Apply Butterworth filtering to the data
    b, a = butter(filter_order, cutoff_freq, fs=1, btype='low', analog=False)
    filtered_y = filtfilt(b, a, y)

    # Store the filtered data in the DataFrame
    data['Filtered Y'] = filtered_y

    # Plot the filtered data and store the line object
    line, = ax.plot(x, filtered_y, label=f'Paddle ({sheet_name})')
    line_visibility.append(line.get_visible())

    # Specify the desired number of windows
    num_windows = 1  # Set the number of windows to a fixed value, e.g., 2

    # Create virtual windows and get index of Max Peak, along with the X and Y values at the end of each window
    window_size = 20 #len(x) // num_windows  # Adjust the window size based on the desired number of windows
    start_x_for_windows = 18  # Windows will start after X=20
    max_peak_idx, end_x_values, end_y_values = get_max_peak_in_windows(x, filtered_y, window_size, num_windows, start_x_for_windows)

    # Store the data for each window in the DataFrame
    temp_peak_data = pd.DataFrame({
        'Sheet': [sheet_name] * len(max_peak_idx),
        'Peak Position(mm)': x[max_peak_idx],
        'Peak Force(N)': filtered_y[max_peak_idx],
        'Steady State Position(mm)': end_x_values,
        'Steady State Force(N)': end_y_values,
        'Window Number': np.arange(1, num_windows + 1)  # Add the window number for each row
    })
    peak_data = pd.concat([peak_data, temp_peak_data], ignore_index=True)

    # Mark the maximum peaks on the plot
    ax.scatter(x[max_peak_idx], filtered_y[max_peak_idx], c='red', marker='x', s=100)
    ax.scatter(end_x_values, end_y_values, c='purple', marker='x', s=100)

# Set plot labels and title
font1 = {'family': 'monospace', 'color': 'Green', 'size': 20, 'fontweight': 'bold'}
font2 = {'family': 'monospace', 'color': 'Green', 'size': 12, 'fontweight': 'bold'}
ax.set_xlabel('Time (s)', fontdict=font2)
ax.set_ylabel('Force (mm)', fontdict=font2)
ax.set_title('Paddle Engagement Graph', fontdict=font1)

# Function to handle legend checkbox updates
def update_legend(label):
    index = legend_labels.index(label)
    line_visibility[index] = not line_visibility[index]
    ax.get_lines()[index].set_visible(line_visibility[index])
    plt.draw()

# Create a list to store the legend labels
legend_labels = [line.get_label() for line in ax.get_lines()]

# Create the CheckButtons instance and connect it to the update_legend function. Turn off if needed.
#checkbox_ax = fig.add_axes([0.1, 0.6, 0.3, 0.3])  # Adjust the position and size as needed
#check_buttons = CheckButtons(checkbox_ax, labels=legend_labels, actives=line_visibility)
#check_buttons.on_clicked(update_legend)

# Add the legend to the figure
ax.legend()

# Show the plot
plt.show()

# Save the peak data and end Y data to an Excel file
output_file_path = '/Users/akin.o.akinjogbin/Desktop/July 2023 Tine Testing Raw data/Tine Engagement /Proccessed Files/Versions Runs/Peak_Data.xlsx'
peak_data.to_excel(output_file_path, index=False)

# Display the peak data for each virtual window, including X and Y values at the end of each window
print("\nPeak Data for Max Peak within Virtual Windows:")
