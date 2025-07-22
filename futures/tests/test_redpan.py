"""
REDPAN Test Suite
================

Comprehensive tests for REDPAN functionality, performance, and compatibility.
"""

import unittest
import numpy as np
import tempfile
import os, sys
import shutil
from pathlib import Path
import tensorflow as tf

# Configure TensorFlow for testing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
tf.config.experimental.enable_op_determinism()

class TestREDPANCore(unittest.TestCase):
    """Test core REDPAN functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pred_npts = 6000
        self.dt = 0.01
        self.batch_size = 4
        self.test_data_length = 18000  # 3 minutes at 100 Hz
        
        # Create synthetic test data
        self.test_waveform = self._create_synthetic_waveform()
        
        # Create temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        tf.keras.backend.clear_session()
    
    def _create_synthetic_waveform(self):
        """Create synthetic 3-component waveform for testing"""
        try:
            from obspy import Stream, Trace, UTCDateTime
            import numpy as np
            
            # Create synthetic data with some realistic seismic characteristics
            t = np.arange(0, self.test_data_length * self.dt, self.dt)
            
            # Add some synthetic P and S arrivals
            p_arrival = 3000  # Sample index
            s_arrival = 5000  # Sample index
            
            # Create 3 components
            traces = []
            for comp in ['Z', 'N', 'E']:
                # Background noise
                data = np.random.normal(0, 0.01, len(t))
                
                # Add P-wave (higher frequency, vertical dominant)
                if comp == 'Z':
                    p_signal = np.exp(-(t - t[p_arrival])**2 / 0.1) * np.sin(2 * np.pi * 10 * t) * 0.1
                    data += p_signal
                
                # Add S-wave (lower frequency, horizontal dominant)  
                if comp in ['N', 'E']:
                    s_signal = np.exp(-(t - t[s_arrival])**2 / 0.2) * np.sin(2 * np.pi * 5 * t) * 0.15
                    data += s_signal
                
                trace = Trace(data=data)
                trace.stats.channel = f'HH{comp}'
                trace.stats.sampling_rate = 1.0 / self.dt
                trace.stats.starttime = UTCDateTime(2023, 1, 1, 0, 0, 0)
                traces.append(trace)
            
            return Stream(traces=traces)
            
        except ImportError:
            # Fallback if obspy not available
            return self._create_numpy_waveform()
    
    def _create_numpy_waveform(self):
        """Create simple numpy array waveform for basic testing"""
        # Simple 3-component data as numpy arrays
        data = []
        for i in range(3):
            component_data = np.random.normal(0, 0.01, self.test_data_length)
            # Add some synthetic signals
            component_data[3000:3100] += 0.1 * np.sin(np.arange(100) * 0.1)
            component_data[5000:5200] += 0.15 * np.sin(np.arange(200) * 0.05)
            data.append(component_data)
        return np.array(data).T
    
    def test_redpan_import(self):
        """Test that REDPAN can be imported successfully"""
        try:
            from redpan import REDPAN, inference_engine
            self.assertTrue(True, "REDPAN imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import REDPAN: {e}")
    
    def test_synthetic_model_creation(self):
        """Test creation of synthetic model for testing"""
        from redpan.core import REDPAN
        
        # Create a minimal synthetic model for testing
        model = self._create_test_model()
        
        # Test REDPAN initialization
        picker = REDPAN(
            model=model,
            pred_npts=self.pred_npts,
            dt=self.dt,
            batch_size=self.batch_size
        )
        
        self.assertIsNotNone(picker)
        self.assertEqual(picker.pred_npts, self.pred_npts)
        self.assertEqual(picker.dt, self.dt)
    
    def _create_test_model(self):
        """Create a minimal TensorFlow model for testing"""
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv1D, Dense
        
        # Simple model that mimics REDPAN architecture
        inputs = Input(shape=(self.pred_npts, 3))
        x = Conv1D(16, 3, activation='relu', padding='same')(inputs)
        x = Conv1D(8, 3, activation='relu', padding='same')(x)
        
        # Three outputs: P, S, and detection mask
        p_output = Conv1D(1, 1, activation='sigmoid', name='p_output')(x)
        s_output = Conv1D(1, 1, activation='sigmoid', name='s_output')(x)
        mask_output = Conv1D(1, 1, activation='sigmoid', name='mask_output')(x)
        
        model = Model(inputs=inputs, outputs=[p_output, s_output, mask_output])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        return model
    
    def test_prediction_shapes(self):
        """Test that predictions have correct shapes"""
        from redpan.core import REDPAN
        
        model = self._create_test_model()
        picker = REDPAN(
            model=model,
            pred_npts=self.pred_npts,
            dt=self.dt,
            batch_size=self.batch_size
        )
        
        # Test with numpy array
        if isinstance(self.test_waveform, np.ndarray):
            predP, predS, predM = picker._predict_numpy(self.test_waveform)
        else:
            # Test with ObsPy Stream
            predP, predS, predM = picker.predict(self.test_waveform)
        
        # Check shapes
        expected_length = self.test_data_length
        self.assertEqual(len(predP), expected_length)
        self.assertEqual(len(predS), expected_length)
        self.assertEqual(len(predM), expected_length)
        
        # Check data types
        self.assertEqual(predP.dtype, np.float32)
        self.assertEqual(predS.dtype, np.float32)
        self.assertEqual(predM.dtype, np.float32)
    
    def test_gaussian_weights(self):
        """Test Gaussian weight generation"""
        from redpan.utils import create_gaussian_weights
        
        weights = create_gaussian_weights(self.pred_npts)
        
        # Check properties
        self.assertEqual(len(weights), self.pred_npts)
        self.assertAlmostEqual(weights.sum(), 1.0, places=3)
        self.assertEqual(np.argmax(weights), self.pred_npts // 2)
    
    def test_memory_efficiency(self):
        """Test memory usage is reasonable"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create picker and process data
        from redpan.core import REDPAN
        model = self._create_test_model()
        picker = REDPAN(
            model=model,
            pred_npts=self.pred_npts,
            dt=self.dt,
            batch_size=self.batch_size
        )
        
        # Process waveform
        if isinstance(self.test_waveform, np.ndarray):
            predP, predS, predM = picker._predict_numpy(self.test_waveform)
        else:
            predP, predS, predM = picker.predict(self.test_waveform)
        
        # Clean up
        del picker, predP, predS, predM
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for this test)
        self.assertLess(memory_increase, 500, 
                       f"Memory usage increased by {memory_increase:.1f}MB")


class TestREDPANCompatibility(unittest.TestCase):
    """Test compatibility with original RED-PAN interfaces"""
    
    def test_factory_function(self):
        """Test factory function interface"""
        try:
            from redpan.factory import inference_engine
            from redpan.models import unets
            
            # This tests the interface, not the actual model loading
            # since we don't have a real model file in tests
            self.assertTrue(callable(inference_engine))
            self.assertTrue(callable(unets))
            
        except ImportError as e:
            self.skipTest(f"Factory test skipped: {e}")
    
    def test_utils_functions(self):
        """Test utility functions"""
        from redpan.utils import create_gaussian_weights, validate_waveform
        
        # Test Gaussian weights
        weights = create_gaussian_weights(1000)
        self.assertEqual(len(weights), 1000)
        
        # Test waveform validation (with numpy array)
        test_data = np.random.normal(0, 1, (10000, 3))
        is_valid = validate_waveform(test_data, min_length=5000)
        self.assertTrue(is_valid)


class TestREDPANPerformance(unittest.TestCase):
    """Performance and benchmark tests"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.test_lengths = [6000, 12000, 60000]  # 1min, 2min, 10min at 100Hz
        
    def test_processing_speed(self):
        """Test processing speed benchmarks"""
        import time
        from redpan.core import REDPAN
        
        # Create test model
        model = self._create_minimal_model()
        picker = REDPAN(model=model, pred_npts=6000, dt=0.01, batch_size=8)
        
        for length in self.test_lengths:
            with self.subTest(length=length):
                # Create test data
                test_data = np.random.normal(0, 0.01, (length, 3))
                
                # Time the prediction
                start_time = time.time()
                predP, predS, predM = picker._predict_numpy(test_data)
                processing_time = time.time() - start_time
                
                # Calculate real-time factor
                waveform_duration = length * 0.01  # seconds
                real_time_factor = waveform_duration / processing_time
                
                # Should be faster than real-time (factor > 1)
                self.assertGreater(real_time_factor, 1.0,
                                 f"Processing too slow for {length} samples: "
                                 f"RTF={real_time_factor:.2f}")
                
                print(f"Length {length}: {processing_time:.3f}s, "
                      f"RTF: {real_time_factor:.1f}x")
    
    def _create_minimal_model(self):
        """Create minimal model for performance testing"""
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv1D
        
        inputs = Input(shape=(6000, 3))
        x = Conv1D(8, 3, activation='relu', padding='same')(inputs)
        
        p_output = Conv1D(1, 1, activation='sigmoid')(x)
        s_output = Conv1D(1, 1, activation='sigmoid')(x)
        mask_output = Conv1D(1, 1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=[p_output, s_output, mask_output])
        return model


class TestREDPANIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    def test_end_to_end_workflow(self):
        """Test complete processing workflow"""
        try:
            # This would test a complete workflow if we had real data and models
            # For now, we'll test the interface
            from redpan import inference_engine
            self.assertTrue(callable(inference_engine))
            
        except ImportError:
            self.skipTest("Integration test requires full REDPAN installation")


def run_all_tests():
    """Run all REDPAN tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestREDPANCore))
    suite.addTests(loader.loadTestsFromTestCase(TestREDPANCompatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestREDPANPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestREDPANIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests when script is executed directly
    success = run_all_tests()
    exit(0 if success else 1)
