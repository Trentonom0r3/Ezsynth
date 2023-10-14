from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import subprocess

# Simulate the ebsynth command
def run_ebsynth(style, source, target, weight, output):
    # Just a placeholder. Normally you'd run the subprocess command here.
    # For example:
    t1 = time.perf_counter()
    subprocess.run([ebsynth_path, "-style", style, "-guide", source, target, "-weight", weight, "-output", output])
    t2 = time.perf_counter()
    #print the command as if it was run in the terminal
    print(f"{ebsynth_path} -style {style} -guide {source} {target} -weight {weight} -output {output}")
    print(f"Time taken: {t2 - t1} seconds")
    pass

# Function to run threads and measure time
def test_threads(num_threads):
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for _ in range(num_threads):
            executor.submit(run_ebsynth, STYLE, SOURCE, TARGET, WEIGHT, OUTPUT)
    end_time = time.time()
    print(f"Time taken with {num_threads} threads: {end_time - start_time} seconds")

# Function to run processes and measure time
def test_processes(num_processes):
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for _ in range(num_processes):
            executor.submit(run_ebsynth, STYLE, SOURCE, TARGET, WEIGHT, OUTPUT)
    end_time = time.time()
    print(f"Time taken with {num_processes} processes: {end_time - start_time} seconds")

if __name__ == "__main__":
    # Constants for ebsynth command
    STYLE = "C:/Users/tjerf/Desktop/Testing/src/Testvids/Intermediate/4.jpg"
    SOURCE = "C:/Users/tjerf/Desktop/Testing/src/Testvids/Input/000.jpg"
    TARGET = "C:/Users/tjerf/Desktop/Testing/src/Testvids/Input/000.jpg"
    WEIGHT = '1.0'
    OUTPUT = "C:/Users/tjerf/Desktop/Testing/src/Testvids/Output/output.jpg"
    ebsynth_path = "C:/Users/tjerf/Desktop/Testing/src/RAFT/ebsynth.exe"
    # Test ThreadPoolExecutor with different number of threads
    t1 = time.perf_counter()
    test_threads(10)
    t2 = time.perf_counter()
    print(f"Time taken 10 threads: {t2 - t1} seconds")
    time.sleep(5)
    t1 = time.perf_counter()
    test_processes(10)
    t2 = time.perf_counter()
    
    