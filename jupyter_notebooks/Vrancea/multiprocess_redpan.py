import sys
import subprocess
import multiprocessing

def run_script(args):
    """Run redpan_pick.py with process_id and total_processes as arguments"""
    process_id, total_processes = args
    command = f"python redpan_pick.py {process_id} {total_processes}"
    process = subprocess.Popen(command, shell=True)
    process.wait()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        num_processes = int(sys.argv[1])
    else:
        num_processes = 3

    # Create argument pairs: (process_id, total_processes)
    args = [(i, num_processes) for i in range(num_processes)]
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(run_script, args)