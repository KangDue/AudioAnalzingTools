import numpy as np
import os
from extract_features import FeatureExtractor
import librosa

def create_test_wav():
    """Create a simple test WAV file"""
    # Generate a test signal (sine wave with some noise)
    duration = 2.0  # seconds
    sample_rate = 48000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a complex signal with multiple frequency components
    signal = (np.sin(2 * np.pi * 100 * t) +  # 100 Hz
              0.5 * np.sin(2 * np.pi * 500 * t) +  # 500 Hz
              0.3 * np.sin(2 * np.pi * 1000 * t) +  # 1000 Hz
              0.1 * np.random.randn(len(t)))  # noise
    
    # Save as WAV file
    test_file = 'test_motor.wav'
    librosa.output.write_wav(test_file, signal, sample_rate)
    return test_file

def test_feature_extraction():
    """Test the feature extraction functionality"""
    print("Testing WAV Analyzer Feature Extraction")
    print("=======================================")
    
    # Create test WAV file
    try:
        test_file = create_test_wav()
        print(f"Created test file: {test_file}")
    except Exception as e:
        print(f"Error creating test file: {e}")
        # Create using scipy instead
        from scipy.io import wavfile
        duration = 2.0
        sample_rate = 48000
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = (np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 500 * t))
        signal = (signal * 32767).astype(np.int16)
        test_file = 'test_motor.wav'
        wavfile.write(test_file, sample_rate, signal)
        print(f"Created test file using scipy: {test_file}")
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Extract features
    print("\nExtracting features...")
    features = extractor.extract_all_features(test_file)
    
    if features:
        print(f"\nFeature extraction successful!")
        print(f"Total features extracted: {len(features) - 1}")  # -1 for filename
        
        # Show feature categories
        wpe_count = len([k for k in features.keys() if k.startswith('WPE')])
        vmd_count = len([k for k in features.keys() if k.startswith('VMD')])
        dcae_count = len([k for k in features.keys() if k.startswith('DCAE')])
        cnn_count = len([k for k in features.keys() if k.startswith('CNN')])
        beam_count = len([k for k in features.keys() if k.startswith('Beamform')])
        src_count = len([k for k in features.keys() if k.startswith('SRC')])
        hht_count = len([k for k in features.keys() if k.startswith('HHT')])
        thermal_count = len([k for k in features.keys() if k.startswith('Thermal')])
        msa_count = len([k for k in features.keys() if k.startswith('MSA')])
        tsa_count = len([k for k in features.keys() if k.startswith('TSA')])
        
        print(f"\nFeature breakdown:")
        print(f"- WPE features: {wpe_count}")
        print(f"- VMD features: {vmd_count}")
        print(f"- DCAE features: {dcae_count}")
        print(f"- CNN features: {cnn_count}")
        print(f"- Beamforming features: {beam_count}")
        print(f"- SRC features: {src_count}")
        print(f"- HHT features: {hht_count}")
        print(f"- Thermal features: {thermal_count}")
        print(f"- MSA features: {msa_count}")
        print(f"- TSA features: {tsa_count}")
        
        total_expected = 8 + 15 + 16 + 12 + 15 + 10 + 10 + 10 + 12 + 10
        total_actual = wpe_count + vmd_count + dcae_count + cnn_count + beam_count + src_count + hht_count + thermal_count + msa_count + tsa_count
        
        print(f"\nExpected total: {total_expected}")
        print(f"Actual total: {total_actual}")
        
        # Show sample feature values
        print(f"\nSample feature values:")
        for i, (key, value) in enumerate(list(features.items())[:10]):
            if key != 'filename':
                print(f"  {key}: {value:.6f}")
        
        print("\n✅ Feature extraction test PASSED!")
        
    else:
        print("❌ Feature extraction test FAILED!")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"\nCleaned up test file: {test_file}")

if __name__ == "__main__":
    test_feature_extraction()