import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import csv
import progressbar
from math import sqrt
from collections import deque
import time

# The function that each core will process independently
def dfs_branch(start_combination, running_sum_init, running_m2_init, n, target_sum_upper, target_sum_lower, target_sd_upper, target_sd_lower, min_scale_sum, max_scale_sum, output_file, n_1, max_scale_1):
    stack = deque([(start_combination, running_sum_init, running_m2_init)])
    results = []
    while stack:
        current, running_sum, running_m2 = stack.pop()
        
        if len(current) >= n:
            current_std = sqrt(running_m2 / n_1)
            if target_sd_lower <= current_std:
                results.append(current)
            continue
        
        n_left = n_1 - len(current)
        next_n = len(current) + 1
        
        for next_value in range(current[-1], max_scale_1):
            # Filter based on mean constraints
            next_sum = running_sum + next_value
            #minmean = next_sum + min_scale_sum[n_left]
            minmean = next_sum + min_scale_sum[next_value-1][n_left]
            if minmean > target_sum_upper:
                break
            maxmean = next_sum + max_scale_sum[n_left]
            if maxmean < target_sum_lower:
                continue
            
            # sd calculations and filter
            next_mean = next_sum / next_n
            delta = next_value - running_sum / len(current)
            delta2 = next_value - next_mean
            next_m2 = running_m2 + delta * delta2
            min_sd = sqrt(next_m2 / n_1)
            if min_sd > target_sd_upper:
                continue
            
            # Push the valid combination with updated stats back into the stack
            stack.append((current + [next_value], next_sum, next_m2))
    
    return results

# Parallel processing part starting from different depths
def parallel_dfs(min_scale, max_scale, n, target_sum, target_sd, rounding_error_sums, rounding_error_sds, num_workers, output_file, depth):
    st = time.time()
    target_sum_upper = target_sum + rounding_error_sums
    target_sum_lower = target_sum - rounding_error_sums
    target_sd_upper = target_sd + rounding_error_sds
    target_sd_lower = target_sd - rounding_error_sds
    #min_scale_sum = [min_scale*n for n in range(n)]
    min_scale_sum = [[min_val * n_left for n_left in range(n)] for min_val in range(min_scale, max_scale+1)]
    max_scale_sum = [max_scale * n_left for n_left in range(n)]
    n_1 = n - 1
    max_scale_1 = max_scale + 1

    # Generate initial combinations of specified depth
    initial_combinations = []
    for comb in itertools.combinations_with_replacement(range(min_scale, max_scale + 1), depth):
        comb_list = list(comb)
        running_sum = sum(comb_list)
        current_mean = running_sum / depth
        running_m2 = sum((x - current_mean) ** 2 for x in comb_list)
        initial_combinations.append((comb_list, running_sum, running_m2))

    # Initialize CSV file and write a header
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'n{i+1}' for i in range(0, n)])

    # Set up progress bar
    widgets = ['Finding combinations: ', progressbar.Bar('*')]
    bar = progressbar.ProgressBar(max_value=len(initial_combinations), widgets=widgets).start()
    progress = 0

    # Parallel execution
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(dfs_branch, combo, running_sum, running_m2, n, target_sum_upper, target_sum_lower, target_sd_upper, target_sd_lower, min_scale_sum, max_scale_sum, output_file, n_1, max_scale_1)
                   for combo, running_sum, running_m2 in initial_combinations]
        
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for future in as_completed(futures):
                progress += 1
                bar.update(progress)
                batch_results = future.result()
                writer.writerows(batch_results)
    
    # Can be necessary for the very fast calculations
    bar.finish()
    # Time taken
    et = time.time() - st
    print(f"Execution time: {et:.2f} seconds")

    # Output number of combinations
    with open(output_file, 'r') as f:
        reader = csv.reader(f)
        print(f"Number of valid combinations: {sum(1 for _ in reader) - 1}")

# Ensure code only runs when called as the main module (necessary for parallel processing)
if __name__ == '__main__':
    # Initialize parameters
    min_scale = 1
    max_scale = 7
    n = 150 # number of people/items
    target_mean = 3
    ### Future optimization: mirror higher numbers to lower ones (which are often faster)
    target_sum = target_mean * n
    target_sd = 1
    rounding_error_means = 0.05
    rounding_error_sums = rounding_error_means * n
    rounding_error_sds = 0.05
    num_workers = cpu_count()
    output_file = 'parallel_results.csv'
    # Depth of parallelization (with a cap at 15)
    depth = min(round(n/10), 15, n-1)

    parallel_dfs(min_scale, max_scale, n, target_sum, target_sd, rounding_error_sums, rounding_error_sds, num_workers, output_file, depth=depth)