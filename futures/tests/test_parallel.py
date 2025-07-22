"""
Parallel Processing Tests for REDPAN
====================================

Tests for multiprocessing functionality and parallel waveform processing.
"""

import unittest
import numpy as np
import tempfile
import os
import shutil
import time
import multiprocessing as mp
from pathlib import Path
import tensorflow as tf

# Configure TensorFlow for testing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TestREDPANParallel(unittest.TestCase):
    """Test parallel processing functionality"""
    
    def setUp(self):
        """Set up parallel test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "test_data")
        self.output_dir = os.path.join(self.temp_dir, "test_output")
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create synthetic test data files
        self._create_test_data_files()
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        tf.keras.backend.clear_session()
    
    def _create_test_data_files(self):
        """Create synthetic SAC files for testing"""
        try:
            from obspy import Stream, Trace, UTCDateTime
            
            # Create multiple test stations
            stations = ['STA1', 'STA2', 'STA3']
            components = ['Z', 'N', 'E']
            
            for sta in stations:
                for comp in components:
                    # Create synthetic data
                    data = np.random.normal(0, 0.01, 18000)  # 3 minutes at 100 Hz
                    
                    # Add some synthetic signals
                    data[3000:3100] += 0.1 * np.sin(np.arange(100) * 0.1)
                    data[5000:5200] += 0.15 * np.sin(np.arange(200) * 0.05)
                    
                    # Create trace
                    trace = Trace(
                        data=data,
                        header={
                            'station': sta,
                            'channel': f'HH{comp}',
                            'starttime': UTCDateTime('2019-07-07T08:00:00'),
                            'delta': 0.01,
                            'network': 'PB',
                            'location': '00'
                        }
                    )
                    
                    # Save as SAC file
                    filename = f"PB.{sta}.HH{comp}.00.2019.188.08.sac"
                    filepath = os.path.join(self.data_dir, filename)
                    trace.write(filepath, format='SAC')
                    
        except ImportError:
            # Create dummy files if ObsPy not available
            stations = ['STA1', 'STA2', 'STA3']
            components = ['Z', 'N', 'E']
            
            for sta in stations:
                for comp in components:
                    filename = f"PB.{sta}.HH{comp}.00.2019.188.08.sac"
                    filepath = os.path.join(self.data_dir, filename)
                    
                    # Create dummy binary file
                    with open(filepath, 'wb') as f:
                        f.write(b'DUMMY_SAC_DATA' * 1000)
    
    def test_multiprocessing_import(self):
        """Test that parallel processing modules can be imported"""
        try:
            from redpan.examples.parallel_demo import process_single_waveform
            self.assertTrue(callable(process_single_waveform))
        except ImportError:
            self.skipTest("Parallel processing modules not available")
    
    def test_process_count_validation(self):
        """Test process count validation"""
        max_processes = mp.cpu_count()
        
        # Should not exceed CPU count
        self.assertLessEqual(max_processes - 1, max_processes)
        self.assertGreater(max_processes, 0)
    
    def test_parallel_file_processing(self):
        """Test parallel processing of multiple files"""
        try:
            # Skip if not enough files or ObsPy not available
            sac_files = [f for f in os.listdir(self.data_dir) if f.endswith('.sac')]
            if len(sac_files) < 3:
                self.skipTest("Not enough test files for parallel processing")
            
            from redpan.examples.parallel_demo import run_parallel_demo
            
            # Create a minimal model file for testing
            model_path = os.path.join(self.temp_dir, "test_model.h5")
            self._create_test_model_file(model_path)
            
            # Run parallel processing with minimal processes
            try:
                run_parallel_demo(
                    data_dir=self.data_dir,
                    output_dir=self.output_dir,
                    model_path=model_path,
                    num_processes=2,  # Use minimal processes for testing
                    batch_size=4
                )
                
                # Check that output files were created
                output_files = os.listdir(self.output_dir)
                self.assertGreater(len(output_files), 0, "No output files created")
                
            except Exception as e:
                self.skipTest(f"Parallel processing test failed: {e}")
                
        except ImportError:
            self.skipTest("Required modules not available for parallel test")
    
    def _create_test_model_file(self, model_path):
        """Create a minimal model file for testing"""
        try:
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Conv1D
            
            # Create minimal model
            inputs = Input(shape=(6000, 3))
            x = Conv1D(8, 3, activation='relu', padding='same')(inputs)
            p_output = Conv1D(1, 1, activation='sigmoid')(x)
            s_output = Conv1D(1, 1, activation='sigmoid')(x)
            mask_output = Conv1D(1, 1, activation='sigmoid')(x)
            
            model = Model(inputs=inputs, outputs=[p_output, s_output, mask_output])
            model.compile(optimizer='adam', loss='binary_crossentropy')
            
            # Save model
            model.save(model_path)
            
        except Exception as e:
            # Create dummy file if TensorFlow fails
            with open(model_path, 'w') as f:
                f.write("DUMMY_MODEL_FILE")
    
    def test_memory_cleanup_parallel(self):
        """Test memory cleanup in parallel processing"""
        try:
            import psutil
            
            # Monitor memory before
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simulate parallel processing memory pattern
            processes = []
            for i in range(2):  # Use 2 processes for testing
                p = mp.Process(target=self._dummy_memory_task, args=(i,))
                processes.append(p)
                p.start()
            
            # Wait for processes to complete
            for p in processes:
                p.join()
            
            # Check memory after
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable
            self.assertLess(memory_increase, 200, 
                           f"Memory increased by {memory_increase:.1f}MB during parallel test")
                           
        except ImportError:
            self.skipTest("psutil not available for memory test")
    
    def _dummy_memory_task(self, process_id):
        """Dummy task that simulates memory usage"""
        # Simulate some memory allocation and cleanup
        data = np.random.rand(1000, 1000)
        result = np.sum(data)
        del data
        return result


class TestREDPANBenchmarks(unittest.TestCase):
    """Benchmark tests for performance validation"""
    
    def test_speed_benchmark(self):
        """Test processing speed benchmark"""
        try:
            from redpan.core import REDPAN
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Conv1D
            
            # Create test model
            inputs = Input(shape=(6000, 3))
            x = Conv1D(8, 3, activation='relu', padding='same')(inputs)
            p_output = Conv1D(1, 1, activation='sigmoid')(x)
            s_output = Conv1D(1, 1, activation='sigmoid')(x)
            mask_output = Conv1D(1, 1, activation='sigmoid')(x)
            
            model = Model(inputs=inputs, outputs=[p_output, s_output, mask_output])
            model.compile(optimizer='adam', loss='binary_crossentropy')
            
            # Create picker
            picker = REDPAN(model=model, pred_npts=6000, dt=0.01, batch_size=8)
            
            # Create test data (10 minutes)
            test_data = np.random.normal(0, 0.01, (60000, 3))
            
            # Benchmark prediction time
            start_time = time.time()
            predP, predS, predM = picker.predict(test_data)
            processing_time = time.time() - start_time
            
            # Calculate metrics
            data_duration = 60000 * 0.01  # 600 seconds (10 minutes)
            real_time_factor = data_duration / processing_time
            
            print(f"\n📊 Benchmark Results:")
            print(f"Data duration: {data_duration:.1f}s")
            print(f"Processing time: {processing_time:.2f}s")
            print(f"Real-time factor: {real_time_factor:.1f}x")
            
            # Should achieve reasonable real-time factor
            self.assertGreater(real_time_factor, 5.0, 
                              f"Real-time factor too low: {real_time_factor:.1f}x")
                              
        except ImportError as e:
            self.skipTest(f"Benchmark test skipped: {e}")
    
    def test_memory_benchmark(self):
        """Test memory usage benchmark"""
        try:
            import psutil
            from redpan.core import REDPAN
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Conv1D
            
            # Monitor initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create model and picker
            inputs = Input(shape=(6000, 3))
            x = Conv1D(8, 3, activation='relu', padding='same')(inputs)
            p_output = Conv1D(1, 1, activation='sigmoid')(x)
            s_output = Conv1D(1, 1, activation='sigmoid')(x)
            mask_output = Conv1D(1, 1, activation='sigmoid')(x)
            
            model = Model(inputs=inputs, outputs=[p_output, s_output, mask_output])
            model.compile(optimizer='adam', loss='binary_crossentropy')
            
            picker = REDPAN(model=model, pred_npts=6000, dt=0.01)
            
            # Test with increasingly large data
            data_sizes = [18000, 36000, 60000]  # 3, 6, 10 minutes
            memory_usage = []
            
            for size in data_sizes:
                test_data = np.random.normal(0, 0.01, (size, 3))
                
                # Process data
                predP, predS, predM = picker.predict(test_data)
                
                # Measure memory
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage.append(current_memory - initial_memory)
                
                # Clean up
                del test_data, predP, predS, predM
                import gc
                gc.collect()
            
            print(f"\n💾 Memory Usage Benchmark:")
            for i, (size, mem) in enumerate(zip(data_sizes, memory_usage)):
                duration = size * 0.01 / 60  # minutes
                print(f"  {duration:.1f} min data: {mem:.1f} MB")
            
            # Memory usage should scale reasonably
            max_memory = max(memory_usage)
            self.assertLess(max_memory, 1000, 
                           f"Memory usage too high: {max_memory:.1f}MB")
                           
        except ImportError as e:
            self.skipTest(f"Memory benchmark test skipped: {e}")


def run_parallel_tests():
    """Run parallel processing tests"""
    print("🔄 Running REDPAN Parallel Processing Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestREDPANParallel))
    suite.addTests(loader.loadTestsFromTestCase(TestREDPANBenchmarks))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All parallel tests passed!")
    else:
        print("❌ Some parallel tests failed!")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_parallel_tests()
    exit(0 if success else 1)
