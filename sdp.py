import os
import itertools
import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np


# Define workloads
workloads = [
    "gem5/SPEC.Small/mcf/mcf",
    "gem5/SPEC.Small/heat/heat",
    "gem5/SPEC.Small/dct/dct",
    "gem5/SPEC.Small/fft/fft",
]

# Replacement mappings based on the user's request
replacements = {
    "l1_cntrl0.east_L1D": "l1_cntrl1.L1Dcache",
    "l1_cntrl0.east_L1I": "l1_cntrl1.L1Icache",
    "l1_cntrl0.south_L1D": "l1_cntrl4.L1Dcache",
    "l1_cntrl0.south_L1I": "l1_cntrl4.L1Icache",
    "l1_cntrl1.east_L1D": "l1_cntrl2.L1Dcache",
    "l1_cntrl1.east_L1I": "l1_cntrl2.L1Icache",
    "l1_cntrl1.south_L1D": "l1_cntrl5.L1Dcache",
    "l1_cntrl1.south_L1I": "l1_cntrl5.L1Icache",
    "l1_cntrl10.south_L1D": "l1_cntrl14.L1Dcache",
    "l1_cntrl10.south_L1I": "l1_cntrl14.L1Icache",
    "l1_cntrl11.south_L1D": "l1_cntrl15.L1Dcache",
    "l1_cntrl11.south_L1I": "l1_cntrl15.L1Icache",
    "l1_cntrl2.east_L1D": "l1_cntrl3.L1Dcache",
    "l1_cntrl2.east_L1I": "l1_cntrl3.L1Icache",
    "l1_cntrl2.south_L1D": "l1_cntrl6.L1Dcache",
    "l1_cntrl2.south_L1I": "l1_cntrl6.L1Icache",
    "l1_cntrl3.south_L1D": "l1_cntrl7.L1Dcache",
    "l1_cntrl3.south_L1I": "l1_cntrl7.L1Icache",
    "l1_cntrl4.south_L1D": "l1_cntrl8.L1Dcache",
    "l1_cntrl4.south_L1I": "l1_cntrl8.L1Icache",
    "l1_cntrl5.south_L1D": "l1_cntrl9.L1Dcache",
    "l1_cntrl5.south_L1I": "l1_cntrl9.L1Icache",
    "l1_cntrl6.south_L1D": "l1_cntrl10.L1Dcache",
    "l1_cntrl6.south_L1I": "l1_cntrl10.L1Icache",
    "l1_cntrl7.south_L1D": "l1_cntrl11.L1Dcache",
    "l1_cntrl7.south_L1I": "l1_cntrl11.L1Icache",
    "l1_cntrl8.south_L1D": "l1_cntrl12.L1Dcache",
    "l1_cntrl8.south_L1I": "l1_cntrl12.L1Icache",
    "l1_cntrl9.south_L1D": "l1_cntrl13.L1Dcache",
    "l1_cntrl9.south_L1I": "l1_cntrl13.L1Icache",
}

old = "old"
new = "new"
config_script = "gem5/configs/deprecated/example/se.py"
num_cpus = "-n 16"
num_dirs = "--num-dirs=16"
mesh_rows = "--mesh-rows=4"
ruby = "--ruby"
network = "--network=garnet"
topology = "--topology=Mesh_XY"
link_latency = "--link-latency=1"
router_latency = "--router-latency=1"
cpu_type = "--cpu-type=O3CPU"
l1d_size = "--l1d_size=32kB"
l1i_size = "--l1i_size=32kB"
l2_size = "--l2_size=4MB"
mem_size = "--mem-size=512MB"
l1d_assoc = "--l1d_assoc=2"
l1i_assoc = "--l1i_assoc=2"
l2_assoc = "--l2_assoc=8"
caches = "--caches"

# Prompt user for the number of simulations to run
core_count = 16
total_combinations = 4 ** core_count
print(f"Total possible combinations: {total_combinations}")
num_simulations = int(input("Enter the number of simulations to run: "))

if num_simulations > total_combinations:
    print("Number of simulations exceeds total possible combinations. Running all simulations.")
    num_simulations = total_combinations



def get_version(version):
    if version == "old":
        gem5_binary = "gem5/build/X86/gem5.opt"
    else:
        gem5_binary = "gem5/build/X86_MESI_Two_Level/gem5.opt"
    return gem5_binary

# Function to generate combinations on-the-fly
def workload_combinations():
    return itertools.product(workloads, repeat=core_count)

def run_simulation(num_simulations, version, load_combinations):
    gem5_bin = get_version(version)
    result_dir_base = f"results/mcpat/{version}"
    ver = f'--version="{version}"'
    
    for sim_index in range(num_simulations):
        combination = next(load_combinations)  # Generate the next combination on demand

        cmd = ";".join(combination)  # Workloads for each core
        result_dir = os.path.join(result_dir_base, f"workload_combination_{sim_index + 1}")
        os.makedirs(result_dir, exist_ok=True)

        command = [
            gem5_bin,
            f'-d {result_dir}',
            config_script,
            f'--cmd="{cmd}"',
            ver,
            num_cpus,
            num_dirs,
            ruby,
            network,
            topology,
            mesh_rows,
            link_latency,
            router_latency,
            cpu_type,
            l1d_size,
            l1i_size,
            l2_size,
            mem_size,
            l1d_assoc,
            l1i_assoc,
            l2_assoc
            
        ]

        print(f"Running simulation for {version} {sim_index + 1}/{num_simulations}...")
        # subprocess.run(" ".join(command), shell=True)
        subprocess.run(" ".join(command), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"Simulation completed for {version}")

def replace(path):
    if not os.path.exists(path):
        print("File path not found")

    # Read the original file, replace the contents, and write back to the file
    with open(path, 'r') as file:
        content = file.read()

    # Perform the replacements
    for old, new in replacements.items():
        content = content.replace(old, new)

    # Write the modified content back to the new file
    with open(path, 'w') as file:
        file.write(content)

def run_parser(config_loc, out_loc, version):
    python = "python3 /home/intel/Cache_simulation_gem5/results/mcpat/Gem5Mcpat_Parser_2024/parser.py"
    config = f"--config {config_loc}config.json"
    stat = f"--stats {config_loc}stats.txt"
    template = "--template /home/intel/Cache_simulation_gem5/results/mcpat/Gem5Mcpat_Parser_2024/templates/template_latest.xml"
    out = f"--output /home/intel/Cache_simulation_gem5/results/mcpat/{version}/configs/out_{out_loc}.xml"

    command = [
        python,
        config,
        stat,
        template,
        out
    ]

    # subprocess.run(" ".join(command), shell=True)
    subprocess.run(" ".join(command), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def mcpat(loc, version):
    mc = f"/home/intel/newmcpat/cMcPAT/mcpat/mcpat"
    infile = f"-infile /home/intel/Cache_simulation_gem5/results/mcpat/{version}/configs/out_{loc}.xml"
    print = "-print_level 5"
    out = f"> /home/intel/Cache_simulation_gem5/results/mcpat/{version}/results/result_{loc}.txt"

    command = [
        mc,
        infile,
        print,
        out
    ]

    # subprocess.run(" ".join(command), shell=True)
    subprocess.run(" ".join(command), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def extract_runtime_dynamic_power(file_path, component_name):
    """
    Extract runtime dynamic power for the specified component from the given file.
    """
    runtime_dynamic = None
    with open(file_path, 'r') as file:
        content = file.read()

        # Define regex patterns for different components
        patterns = {
            "Processor": r"Processor:\s+.*?Runtime Dynamic = (\d+\.\d+) W",
            "Core": r"Total Cores:.*?Runtime Dynamic = (\d+\.\d+) W",
            "L2 Cache": r"Total L2s:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Bus": r"Total NoCs \(Network/Bus\):.*?Runtime Dynamic = (\d+\.\d+) W",
            "Instruction Fetch Unit": r"Instruction Fetch Unit:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Instruction Cache": r"Instruction Cache:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Branch Target Buffer": r"Branch Target Buffer:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Branch Predictor": r"Branch Predictor:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Global Predictor": r"Global Predictor:.*?Runtime Dynamic = (\d+\.\d+) W",
            "L1_Local Predictor": r"L1_Local Predictor:.*?Runtime Dynamic = (\d+\.\d+) W",
            "L2_Local Predictor": r"L2_Local Predictor:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Chooser": r"Chooser:.*?Runtime Dynamic = (\d+\.\d+) W",
            "RAS": r"RAS:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Instruction Buffer": r"Instruction Buffer:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Instruction Decoder": r"Instruction Decoder:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Renaming Unit": r"Renaming Unit:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Int Front End RAT": r"Int Front End RAT.*?Runtime Dynamic = (\d+\.\d+) W",
            "FP Front End RAT": r"FP Front End RAT.*?Runtime Dynamic = (\d+\.\d+) W",
            "Free List": r"Free List:.*?Runtime Dynamic = (\d+\.\d+) W",
            "FP Free List": r"FP Free List:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Load Store Unit": r"Load Store Unit:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Data Cache": r"Data Cache:.*?Runtime Dynamic = (\d+\.\d+) W",
            "LoadQ": r"LoadQ:.*?Runtime Dynamic = (\d+\.\d+) W",
            "StoreQ": r"StoreQ:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Memory Management Unit": r"Memory Management Unit:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Itlb": r"Itlb:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Dtlb": r"Dtlb:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Execution Unit": r"Execution Unit:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Register Files": r"Register Files:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Integer RF": r"Integer RF:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Floating Point RF": r"Floating Point RF:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Instruction Scheduler": r"Instruction Scheduler:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Instruction Window": r"Instruction Window:.*?Runtime Dynamic = (\d+\.\d+) W",
            "FP Instruction Window": r"FP Instruction Window:.*?Runtime Dynamic = (\d+\.\d+) W",
            "ROB": r"ROB:.*?Runtime Dynamic = (\d+\.\d+) W",
            "Integer ALUs": r"Integer ALUs.*?Runtime Dynamic = (\d+\.\d+) W",
            "Floating Point Units": r"Floating Point Units.*?Runtime Dynamic = (\d+\.\d+) W",
            "Complex ALUs": r"Complex ALUs.*?Runtime Dynamic = (\d+\.\d+) W",
            "Results Broadcast Bus": r"Results Broadcast Bus:.*?Runtime Dynamic = (\d+\.\d+) W"
        }

        # Match and extract runtime dynamic power
        if component_name in patterns:
            match = re.search(patterns[component_name], content, re.DOTALL)
            if match:
                runtime_dynamic = float(match.group(1))
    return runtime_dynamic

def process_files_in_directory(folder_path, component_name):
    """
    Process all result files in the directory and collect runtime dynamic power.
    """
    results = {}
    for file_name in os.listdir(folder_path):
        if file_name.startswith("result_") and file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            power = extract_runtime_dynamic_power(file_path, component_name)
            if power is not None:
                results[file_name] = power
    return results

def plot_runtime_dynamic_power(results_1, results_2, component_name, plot_path):
    """
    Plot the runtime dynamic power for the given component side by side from two directories.
    """
    file_names = sorted(set(results_1.keys()) | set(results_2.keys()))
    powers_1 = [results_1.get(name, 0) for name in file_names]
    powers_2 = [results_2.get(name, 0) for name in file_names]

    x = np.arange(len(file_names))  # the label locations
    width = 0.35  # the width of the bars

    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width/2, powers_1, width, label='old', color='orange')
    bars2 = plt.bar(x + width/2, powers_2, width, label='new', color='lightgreen')

    # Add value labels on top of each bar
    for bar, power in zip(bars1, powers_1):
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # X position
            bar.get_height() / 2,  # Y position
            f'{power}',  # Value (formatted to 2 decimal places)
            ha='center',  # Horizontal alignment
            va='bottom',  # Vertical alignment
            rotation='vertical'  # Rotate text vertically
        )
    for bar, power in zip(bars2, powers_2):
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # X position
            bar.get_height() / 2,  # Y position
            f'{power}',  # Value (formatted to 2 decimal places)
            ha='center',  # Horizontal alignment
            va='bottom',  # Vertical alignment
            rotation='vertical'  # Rotate text vertically
        )

    plt.xticks(x, file_names, rotation=45, ha="right")
    plt.ylabel("Runtime Dynamic Power (W)")
    plt.title(f"Runtime Dynamic Power of {component_name} (Side by Side Comparison)")
    plt.legend()
    plt.tight_layout()
     # Save the plot as an image
    plt.savefig(plot_path)
    # print(f"Plot saved")
    # plt.show()




combination_generator = workload_combinations()
run_simulation(num_simulations, old, combination_generator)
run_simulation(num_simulations, new, combination_generator)



for i in range(num_simulations):
    file_path = f'/home/intel/Cache_simulation_gem5/results/mcpat/new/workload_combination_{i+1}/stats.txt'
    replace(file_path)
print("Replaced result stats")

for i in range(num_simulations):
    loc = f"/home/intel/Cache_simulation_gem5/results/mcpat/{old}/workload_combination_{i+1}/"
    run_parser(loc, i+1, old)

    loc = f"/home/intel/Cache_simulation_gem5/results/mcpat/{new}/workload_combination_{i+1}/"
    run_parser(loc, i+1, new)
print("Parser completed")


print("McPat simulation started")
for i in range(num_simulations):
    mcpat(i+1, old)
    mcpat(i+1, new)
print("McPat simulation completed")

print("Saving Plots")
components = [
    "Processor",
    "Core",
    "L2 Cache",
    "Bus",
    "Instruction Cache",

]
folder_path_1 = "/home/intel/Cache_simulation_gem5/results/mcpat/old/results/"
folder_path_2 = "/home/intel/Cache_simulation_gem5/results/mcpat/new/results/"

for component_name in components:
    plot_path = f"/home/intel/Cache_simulation_gem5/results/mcpat/plots/plot_{component_name}.png"
    results_1 = process_files_in_directory(folder_path_1, component_name)
    results_2 = process_files_in_directory(folder_path_2, component_name)
    plot_runtime_dynamic_power(results_1, results_2, component_name, plot_path)
print("Saved Plots")


print("All simulations completed.")
