"""
Command Line Interface for REDPAN
"""

import argparse
import sys
import os
import multiprocessing as mp
from pathlib import Path

def benchmark_command():
    """Run performance benchmarks"""
    print("Running REDPAN benchmarks...")
    # Import here to avoid circular imports
    from redpan.benchmarks import run_benchmarks
    run_benchmarks()

def test_command():
    """Run test suite"""
    print("Running REDPAN tests...")
    import pytest
    test_dir = Path(__file__).parent.parent / "tests"
    pytest.main([str(test_dir), "-v"])

def demo_command():
    """Run REDPAN demo"""
    parser = argparse.ArgumentParser(description="Run REDPAN demo")
    parser.add_argument("--data-dir", required=True, help="Directory containing waveform data")
    parser.add_argument("--output-dir", required=True, help="Output directory for picks")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    parser.add_argument("--processes", type=int, default=mp.cpu_count()-1, help="Number of processes")
    
    args = parser.parse_args()
    
    from redpan.examples.demo import run_demo
    run_demo(
        data_dir=args.data_dir,
        output_dir=args.output_dir, 
        model_path=args.model,
        parallel=args.parallel,
        num_processes=args.processes
    )

def parallel_command():
    """Run parallel processing demo"""
    parser = argparse.ArgumentParser(description="Run REDPAN parallel processing")
    parser.add_argument("--data-dir", required=True, help="Directory containing waveform data")
    parser.add_argument("--output-dir", required=True, help="Output directory for picks")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--processes", type=int, default=mp.cpu_count()-1, help="Number of processes")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for processing")
    
    args = parser.parse_args()
    
    from redpan.examples.parallel_demo import run_parallel_demo
    run_parallel_demo(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_path=args.model,
        num_processes=args.processes,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Available commands: benchmark, test, demo, parallel")
        sys.exit(1)
    
    command = sys.argv[1]
    if command == "benchmark":
        benchmark_command()
    elif command == "test":
        test_command()
    elif command == "demo":
        demo_command()
    elif command == "parallel":
        parallel_command()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
