from queue import LifoQueue
import time
from math import sqrt
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import csv
import progressbar

# The function that each core will process independently
def dfs_branch(start_combination, running_sum_init, running_m2_init, n, target_sum_upper, target_sum_lower, target_sd_upper, target_sd_lower, min_scale_sum, max_scale_sum, output_file, n_1, max_scale_1):
    stack = LifoQueue()
    stack.put((start_combination, running_sum_init, running_m2_init))
    results = []  # Collect valid combinations in memory
    with open(output_file, 'a', newline='') as f:
        #writer = csv.writer(f)
        while not stack.empty():
            current, running_sum, running_m2 = stack.get()
            
            if len(current) >= n:
                current_std = sqrt(running_m2/n_1)#statistics.stdev(current)
                if target_sd_lower <= current_std:
                    results.append(current)
                    #with lock:  # Lock for safe file write
                    #    writer.writerow(current)
                        #f.write(",".join(map(str, current)) + "\n")
                continue
            n_left = (n_1 - len(current))
            next_n = len(current) + 1

            for next_value in range(current[-1], max_scale_1):
                # Filter based on mean constraints
                next_sum = running_sum + next_value
                minmean = next_sum + min_scale_sum[n_left]
                if minmean > target_sum_upper:
                    continue
                maxmean = next_sum + max_scale_sum[n_left]
                if maxmean < target_sum_lower:
                    continue
                
                # sd calculations and filter
                next_mean = next_sum / next_n
                delta = next_value - running_sum / len(current)
                delta2 = next_value - next_mean
                next_m2 = running_m2 + delta * delta2
                min_sd = sqrt(next_m2 / (n_1))
                if min_sd > target_sd_upper:
                    continue

                # Push the valid combination with updated stats back into the stack
                stack.put((current + [next_value], next_sum, next_m2))

        return results

# Parallel processing part starting from depth n=2
def parallel_dfs(min_scale, max_scale, n, target_sum, target_sd, rounding_error_sums, rounding_error_sds, num_workers, output_file):
    st = time.time()
    target_sum_upper = target_sum + rounding_error_sums
    target_sum_lower = target_sum - rounding_error_sums
    target_sd_upper = target_sd + rounding_error_sds
    target_sd_lower = target_sd - rounding_error_sds
    min_scale_sum = [min_scale*n for n in range(n)]
    max_scale_sum = [max_scale*n for n in range(n)]
    n_1 = n - 1
    max_scale_1 = max_scale + 1

    # Generate initial combinations for depth 3
    initial_combinations = []
    for i in range(min_scale, max_scale + 1):
        for j in range(i, max_scale + 1):
            initial_combination = [i, j]
            running_sum = sum(initial_combination)
            current_mean = running_sum/2
            current_m2 = (i - current_mean)**2 + (j - current_mean)**2
            initial_combinations.append((initial_combination, running_sum, current_m2))

    # Initialize the CSV file and write a header
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'n{i+1}' for i in range(0,n)])

    # Initialize a lock to synchronize file writes between processes
    #with Manager() as manager:
    #    lock = manager.Lock()

    # Progress bar set up
    widgets = ['Finding combinations: ',
           progressbar.Bar('*'),
          ]
    bar = progressbar.ProgressBar(max_value=len(initial_combinations),
                                  widgets=widgets).start()
    progress = 0

    # Use ProcessPoolExecutor to parallelize DFS
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(dfs_branch, combo, running_sum, running_m2, n, target_sum_upper, target_sum_lower, target_sd_upper, target_sd_lower, min_scale_sum, max_scale_sum, output_file, n_1, max_scale_1) 
                for combo, running_sum, running_m2, in initial_combinations]
        
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for future in as_completed(futures):
                progress +=1
                bar.update(progress)
                batch_results = future.result()
                writer.writerows(batch_results)

            #for future in as_completed(futures):
            #    future.result()
    
    # Can be necessary for the very fast calculations
    bar.update(progress)
    # Time taken
    et = time.time() - st
    print(f"Execution time: {et:.2f} seconds")

    # Output number of combinations
    with open(output_file, 'r') as f:
        reader = csv.reader(f)
        print(f"Number of valid combinations: {sum(1 for _ in reader)-1}")

# Ensure code only runs when called as the main module (necessary for parallel processing)
if __name__ == '__main__':
    # Initialize parameters
    min_scale = 1
    max_scale = 7
    n = 30 # number of people/items
    target_mean = 5
    target_sum = target_mean * n
    target_sd = 2.78
    rounding_error_means = 0.01
    rounding_error_sums = rounding_error_means * n
    rounding_error_sds = 0.01
    num_workers = cpu_count()  # Number of CPU cores to use
    output_file = 'parallel_results.csv'

    # Run parallel DFS
    parallel_dfs(min_scale, max_scale, n, target_sum, target_sd, rounding_error_sums, rounding_error_sds, num_workers, output_file)

