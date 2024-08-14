import random
import xlwings as xw
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, kurtosis
import pandas as pd

# Specify the directory and file name of the Excel workbook
excel_file_path = '/Users/akin.o.akinjogbin/Desktop/TS folder/Proximal End Adaptor _GB-1 Lead TS.xlsx'  # Replace with the actual directory and file path
sheet_name = 'TS-02_Proposed_cone tip'

# Check if Excel is already open, and if not, create a new instance
if len(xw.apps.keys()) == 0:
    app = xw.App(visible=False)
else:
    app = xw.apps.active

# Open the Excel file
workbook = app.books.open(excel_file_path)
worksheet = workbook.sheets[sheet_name]

# Function to extract values from Excel formulas and convert to float
def get_excel_formula_value(cell_address):
    cell = worksheet.range(cell_address)
    if cell.formula is not None:
        # If the cell contains a formula, evaluate the formula to get the result
        return float(cell.value) if cell.value is not None else None
    else:
        # If not a formula, return the cell's value as a float if it's numeric
        cell_value = cell.value
        if cell_value is not None and isinstance(cell_value, (int, float)):
            return float(cell_value)
        else:
            return None

# Specify the cell addresses for nominal values and tolerances
component1_nominal = 'D25'
component1_tolerance = 'E25'
component2_nominal = 'D26'
component2_tolerance = 'E26'


# Extract values from the specified cells using the function and convert to float
component1_nominal = get_excel_formula_value(component1_nominal)
component1_tolerance = get_excel_formula_value(component1_tolerance)

component2_nominal = get_excel_formula_value(component2_nominal)
component2_tolerance = get_excel_formula_value(component2_tolerance)

# Read the threshold values from Excel cells
threshold_start_cell = 'J4'
threshold_end_cell = 'J5'

# Get the threshold values from Excel cells
threshold_start = get_excel_formula_value(threshold_start_cell)
threshold_end = get_excel_formula_value(threshold_end_cell)

# Define the number of Monte Carlo iterations
num_iterations = 1000

# Define the number of Monte Carlo simulations
num_simulations = 50  # Change this to the number of desired simulations

# Initialize lists to store output values for each simulation
all_actual_simulations = []
all_min_gap = []
all_max_gap = []
all_nominal_gap = []
all_median_gap = []
all_mean_gap = []
all_ci_high = []
all_ci_low = []
all_sigma = []
all_variance_gap = []
all_kurtosis_gap = []
all_l_out_percentage = []
all_h_out_percentage = []
all_total_out_percentage = []
all_exceedance = []

# Run the specified number of Monte Carlo simulations
for simulation in range(num_simulations):
    gap_sizes = []  # Initialize a list to store gap sizes for this simulation
    exceedance = 0  # Initialize the count of exceedances for this simulation
    
    for _ in range(num_iterations):
        # Generate random samples for each component based on their tolerances
        component1_sample = random.uniform(component1_nominal - component1_tolerance, component1_nominal + component1_tolerance)
        component2_sample = random.uniform(component2_nominal - component2_tolerance, component2_nominal + component2_tolerance)
        
        # Calculate the gap size between the components
        gap_size = component1_sample - component2_sample
        
        # Append the gap size to the results list for this simulation
        gap_sizes.append(gap_size)

        # Check if the gap size falls outside the specified threshold range
        if gap_size < threshold_start or gap_size > threshold_end:
            exceedance += 1

    # Calculate statistics for this simulation
    actual_simulations = len(gap_sizes)
    min_gap = min(gap_sizes)
    max_gap = max(gap_sizes)
    nominal_gap = component1_nominal - component2_nominal
    median_gap = np.median(gap_sizes)
    mean_gap = np.mean(gap_sizes)
    sigma = np.std(gap_sizes)
    range_gap = max_gap - min_gap

    # Calculate 95% confidence intervals for the mean
    z_critical = norm.ppf(0.975)
    margin_error = z_critical * (sigma / np.sqrt(actual_simulations))
    ci_low = mean_gap - margin_error
    ci_high = mean_gap + margin_error

    # Calculate variance and kurtosis using NumPy and scipy.stats
    variance_gap = np.var(gap_sizes)
    kurtosis_gap = kurtosis(gap_sizes)

    # Append the results for this simulation to the lists
    all_actual_simulations.append(actual_simulations)
    all_min_gap.append(min_gap)
    all_max_gap.append(max_gap)
    all_nominal_gap.append(nominal_gap)
    all_median_gap.append(median_gap)
    all_mean_gap.append(mean_gap)
    all_ci_high.append(ci_high)
    all_ci_low.append(ci_low)
    all_sigma.append(sigma)
    all_variance_gap.append(variance_gap)
    all_kurtosis_gap.append(kurtosis_gap)
    #all_l_out_percentage.append(l_out_percentage)
    #all_h_out_percentage.append(h_out_percentage)
    #all_total_out_percentage.append(total_out_percentage)
    all_exceedance.append(exceedance)

    # Print the simulation number as it progresses
    print(f"Simulation {simulation + 1}/{num_simulations} completed.")

# Calculate the averages of the results from all simulations
average_actual_simulations = np.mean(all_actual_simulations)
average_min_gap = np.mean(all_min_gap)
average_max_gap = np.mean(all_max_gap)
average_nominal_gap = np.mean(all_nominal_gap)
average_median_gap = np.mean(all_median_gap)
average_mean_gap = np.mean(all_mean_gap)
average_ci_high = np.mean(all_ci_high)
average_ci_low = np.mean(all_ci_low)
average_sigma = np.mean(all_sigma)
average_variance_gap = np.mean(all_variance_gap)
average_kurtosis_gap = np.mean(all_kurtosis_gap)
average_l_out_percentage = np.mean(all_l_out_percentage)
average_h_out_percentage = np.mean(all_h_out_percentage)
average_total_out_percentage = np.mean(all_total_out_percentage)
average_exceedance = np.mean(all_exceedance)

# Create a DataFrame to store the average results
average_results = pd.DataFrame({
    'Output': ['Total Simulated Runs', 'Minimum Gap(mm)', 'Maximum Gap(mm)', 'Nominal Gap(mm)',
               'Median Gap(mm)', 'Mean Gap(mm)', '95% Confidence Interval (High)(mm)', '95% Confidence Interval  (Low)(mm)', 'Standard Deviation(mm)',
               'Variance (mm^2)', 'Kurtosis', 'Number of Simulations', 'Exceedance'],
    'Value': [average_actual_simulations, average_min_gap, average_max_gap, average_nominal_gap, average_median_gap,
              average_mean_gap, average_ci_high, average_ci_low, average_sigma, average_variance_gap, average_kurtosis_gap,
              num_simulations, average_exceedance]
})

# Save the average results to an Excel file
average_results.to_excel(f'{sheet_name} monte_carlo_average_results.xlsx', index=False)



# Create a histogram of the gap size results
plt.hist(gap_sizes, bins=30, color='blue', alpha=0.7, edgecolor='black', density=True)
          

# Plot exceedance threshold lines
plt.axvline(threshold_start, color='red', linestyle='dashed', linewidth=2, label=f'Threshold Start ({threshold_start})')
plt.axvline(threshold_end, color='green', linestyle='dashed', linewidth=2, label=f'Threshold End ({threshold_end})')

# Fit a bell-shaped curve (normal distribution) to the data
mu, std = norm.fit(gap_sizes)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label=f'Normal Distribution: $\mu={mu:.2f}$, $\sigma={std:.2f}$')


plt.legend()
plt.title('Monte Carlo Distribution Chart')
plt.xlabel('Gap Size')
plt.ylabel('Density')
plt.grid(True)

# Show the combined plots
plt.show()

workbook.save()
# workbook.close()
# app.quit()
