import os
import re
import itertools
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# Configuration Parameters
WORKLOADS = [
    "gem5/SPEC.Small/mcf/mcf",
    "gem5/SPEC.Small/heat/heat",
    "gem5/SPEC.Small/dct/dct",
    "gem5/SPEC.Small/fft/fft",
]
REPLACEMENTS = {
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
COMPONENTS = [
    "Processor",
    "Core",
    "L2 Cache",
    "Bus",
    "Instruction Cache",
]
GEM5_CONFIG = {
    "num_cpus": "-n 16",
    "num_dirs": "--num-dirs=16",
    "mesh_rows": "--mesh-rows=4",
    "ruby": "--ruby",
    "network": "--network=garnet",
    "topology": "--topology=Mesh_XY",
    "link_latency": "--link-latency=1",
    "router_latency": "--router-latency=1",
    "cpu_type": "--cpu-type=O3CPU",
    "l1d_size": "--l1d_size=32kB",
    "l1i_size": "--l1i_size=32kB",
    "l2_size": "--l2_size=4MB",
    "mem_size": "--mem-size=512MB",
    "l1d_assoc": "--l1d_assoc=2",
    "l1i_assoc": "--l1i_assoc=2",
    "l2_assoc": "--l2_assoc=8",
}
GEM5_BINARY_OLD = "gem5/build/X86/gem5.opt"
GEM5_BINARY_NEW = "gem5/build/X86_MESI_Two_Level/gem5.opt"
GEM5_CONFIG_SCRIPT = "gem5/configs/deprecated/example/se.py"
RESULTS_PATH = "/home/intel/Cache_simulation_gem5/results/mcpat"

# Utility Functions
def workload_combinations(workloads, core_count):
    return itertools.product(workloads, repeat=core_count)

def replace_file_content(file_path, replacements):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
    with open(file_path, 'r') as file:
        content = file.read()
    for old, new in replacements.items():
        content = content.replace(old, new)
    with open(file_path, 'w') as file:
        file.write(content)

def extract_runtime_dynamic_power(file_path, component_name):
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
    if component_name not in patterns:
        print(f"Component {component_name} not recognized.")
        return None
    with open(file_path, 'r') as file:
        content = file.read()
    match = re.search(patterns[component_name], content, re.DOTALL)
    return float(match.group(1)) if match else None

def process_files_in_directory(folder_path, component_name):
    results = {}
    for file_name in os.listdir(folder_path):
        if file_name.startswith("result_") and file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            power = extract_runtime_dynamic_power(file_path, component_name)
            if power is not None:
                results[file_name] = power
    return results

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


# Simulation Functions
def run_gem5_simulations(workloads, version, num_simulations):
    gem5_binary = GEM5_BINARY_OLD if version == "old" else GEM5_BINARY_NEW
    results_dir = f"{RESULTS_PATH}/{version}"
    ver = f'--version="{version}"'
    os.makedirs(results_dir, exist_ok=True)
    combinations = workload_combinations(workloads, 16)
    for i in range(num_simulations):
        combination = next(combinations)
        result_dir = os.path.join(results_dir, f"workload_combination_{i+1}")
        os.makedirs(result_dir, exist_ok=True)
        cmd = ";".join(combination)
        print(GEM5_CONFIG["cpu_type"])
        command = [
            gem5_binary,
            f"-d {result_dir}",
            GEM5_CONFIG_SCRIPT,
            f'--cmd="{cmd}"',
            ver,
            GEM5_CONFIG["cpu_type"],
            GEM5_CONFIG["num_cpus"],
            GEM5_CONFIG["num_dirs"],
            GEM5_CONFIG["ruby"],
            GEM5_CONFIG["network"],
            GEM5_CONFIG["topology"],
            GEM5_CONFIG["mesh_rows"],
            GEM5_CONFIG["link_latency"],
            GEM5_CONFIG["router_latency"],
            GEM5_CONFIG["l1d_size"],
            GEM5_CONFIG["l1i_size"],
            GEM5_CONFIG["l2_size"],
            GEM5_CONFIG["mem_size"],
            GEM5_CONFIG["l1d_assoc"],
            GEM5_CONFIG["l1i_assoc"],
            GEM5_CONFIG["l2_assoc"]
            
            
        ]
        subprocess.run(" ".join(command), shell=True)

def run_mcpat(version, i):
    mc = "/home/intel/newmcpat/cMcPAT/mcpat/mcpat"
    infile = f"-infile {RESULTS_PATH}/{version}/configs/out_{i}.xml"
    output_file = f"> {RESULTS_PATH}/{version}/results/result_{i}.txt"
    command = [mc, infile, "-print_level 5", output_file]
    subprocess.run(" ".join(command), shell=True)

# Plotting
# def plot_runtime_dynamic_power(results_1, results_2, component_name, output_path):
#     file_names = sorted(set(results_1.keys()) | set(results_2.keys()))
#     powers_1 = [results_1.get(name, 0) for name in file_names]
#     powers_2 = [results_2.get(name, 0) for name in file_names]

#     x = np.arange(len(file_names))
#     width = 0.35

#     bars1 = plt.bar(x - width/2, powers_1, width, label="Old", color="orange")
#     bars2 = plt.bar(x + width/2, powers_2, width, label="New", color="green")

#     # Add value labels on top of each bar
#     for bar, power in zip(bars1, powers_1):
#         plt.text(
#             bar.get_x() + bar.get_width() / 2,  # X position
#             bar.get_height() / 2,  # Y position
#             f'{power}',  # Value (formatted to 2 decimal places)
#             ha='center',  # Horizontal alignment
#             va='bottom',  # Vertical alignment
#             rotation='vertical'  # Rotate text vertically
#         )
#     for bar, power in zip(bars2, powers_2):
#         plt.text(
#             bar.get_x() + bar.get_width() / 2,  # X position
#             bar.get_height() / 2,  # Y position
#             f'{power}',  # Value (formatted to 2 decimal places)
#             ha='center',  # Horizontal alignment
#             va='bottom',  # Vertical alignment
#             rotation='vertical'  # Rotate text vertically
#         )

#     plt.xticks(x, file_names, rotation=45, ha="right")
#     plt.ylabel("Runtime Dynamic Power (W)")
#     plt.title(f"Runtime Dynamic Power: {component_name}")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(output_path)
#     # plt.show()

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



# Main Script
def main():
    core_count = 16
    num_simulations = int(input("Enter the number of simulations to run: "))
    # run_gem5_simulations(WORKLOADS, "old", num_simulations)
    # run_gem5_simulations(WORKLOADS, "new", num_simulations)

    for i in range(num_simulations):
        file_path = f'/home/intel/Cache_simulation_gem5/results/mcpat/new/workload_combination_{i+1}/stats.txt'
        replace_file_content(file_path, REPLACEMENTS)

    for i in range(num_simulations):
        loc = f"/home/intel/Cache_simulation_gem5/results/mcpat/old/workload_combination_{i+1}/"
        run_parser(loc, i+1, "old")

        loc = f"/home/intel/Cache_simulation_gem5/results/mcpat/new/workload_combination_{i+1}/"
        run_parser(loc, i+1, "new")
    print("Parser completed")

    # for i in range(num_simulations):
    #     run_mcpat("old", i+1)
    #     run_mcpat("new", i+1)
    print("McPat simulation completed")


    for component in COMPONENTS:
        results_old = process_files_in_directory(f"{RESULTS_PATH}/old/results", component)
        results_new = process_files_in_directory(f"{RESULTS_PATH}/new/results", component)
        plot_runtime_dynamic_power(
            results_old, results_new, component, f"{RESULTS_PATH}/plots/{component}.png"
        )

if __name__ == "__main__":
    main()
