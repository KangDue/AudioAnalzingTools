import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from extract_features import FeatureExtractor

def create_demo_files():
    """Create demo WAV files for testing"""
    print("Creating demo WAV files...")
    
    # Create demo folder
    demo_folder = "demo_wav_files"
    os.makedirs(demo_folder, exist_ok=True)
    
    sample_rate = 48000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Normal motor sound (low frequency, stable)
    normal_signal = (np.sin(2 * np.pi * 60 * t) +  # 60 Hz fundamental
                    0.3 * np.sin(2 * np.pi * 120 * t) +  # 2nd harmonic
                    0.1 * np.sin(2 * np.pi * 180 * t) +  # 3rd harmonic
                    0.05 * np.random.randn(len(t)))  # low noise
    
    # Faulty motor sound (higher frequency, more noise, impacts)
    fault_signal = (np.sin(2 * np.pi * 60 * t) +  # 60 Hz fundamental
                   0.5 * np.sin(2 * np.pi * 120 * t) +  # stronger 2nd harmonic
                   0.3 * np.sin(2 * np.pi * 180 * t) +  # stronger 3rd harmonic
                   0.8 * np.sin(2 * np.pi * 1200 * t) +  # high freq fault
                   0.2 * np.random.randn(len(t)))  # more noise
    
    # Add periodic impacts to fault signal
    impact_freq = 25  # 25 Hz impact frequency
    impact_times = np.arange(0, duration, 1/impact_freq)
    for impact_time in impact_times:
        impact_idx = int(impact_time * sample_rate)
        if impact_idx < len(fault_signal):
            # Add exponentially decaying impact
            impact_duration = 0.01  # 10ms impact
            impact_samples = int(impact_duration * sample_rate)
            impact_envelope = np.exp(-np.arange(impact_samples) / (impact_samples/5))
            impact_signal = 2.0 * impact_envelope * np.sin(2 * np.pi * 2000 * np.arange(impact_samples) / sample_rate)
            
            end_idx = min(impact_idx + impact_samples, len(fault_signal))
            fault_signal[impact_idx:end_idx] += impact_signal[:end_idx-impact_idx]
    
    # Normalize and convert to int16
    normal_signal = (normal_signal / np.max(np.abs(normal_signal)) * 0.8 * 32767).astype(np.int16)
    fault_signal = (fault_signal / np.max(np.abs(fault_signal)) * 0.8 * 32767).astype(np.int16)
    
    # Save files
    normal_file = os.path.join(demo_folder, "motor_normal.wav")
    fault_file = os.path.join(demo_folder, "motor_fault.wav")
    
    wavfile.write(normal_file, sample_rate, normal_signal)
    wavfile.write(fault_file, sample_rate, fault_signal)
    
    print(f"Created: {normal_file}")
    print(f"Created: {fault_file}")
    
    return demo_folder

def analyze_demo_files(folder_path):
    """Analyze demo files and save results"""
    print(f"\nAnalyzing WAV files in: {folder_path}")
    
    # Find WAV files
    wav_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith('.wav'):
            wav_files.append(os.path.join(folder_path, file))
    
    if not wav_files:
        print("No WAV files found!")
        return
    
    print(f"Found {len(wav_files)} WAV files")
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Extract features
    all_features = []
    
    for wav_file in wav_files:
        print(f"Processing: {os.path.basename(wav_file)}")
        features = extractor.extract_all_features(wav_file)
        if features is not None:
            features['filename'] = os.path.basename(wav_file)
            all_features.append(features)
    
    if not all_features:
        print("No features extracted!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Reorder columns
    cols = ['filename'] + [col for col in df.columns if col != 'filename']
    df = df[cols]
    
    # Save to CSV
    output_path = os.path.join(folder_path, 'features_extracted.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Analysis completed!")
    print(f"Results saved to: {output_path}")
    print(f"Extracted {len(df.columns)-1} features from {len(df)} files")
    
    # Show comparison
    print(f"\n📊 Feature Comparison:")
    for idx, row in df.iterrows():
        filename = row['filename']
        print(f"\n{filename}:")
        
        # Show some key features
        wpe_mean = np.mean([row[col] for col in df.columns if col.startswith('WPE')])
        vmd_mean = np.mean([row[col] for col in df.columns if col.startswith('VMD')])
        hht_mean = np.mean([row[col] for col in df.columns if col.startswith('HHT')])
        
        print(f"  WPE avg: {wpe_mean:.4f}")
        print(f"  VMD avg: {vmd_mean:.4f}")
        print(f"  HHT avg: {hht_mean:.4f}")
    
    return output_path

def main():
    print("WAV Analyzer Demo")
    print("=================")
    
    # Create demo files
    demo_folder = create_demo_files()
    
    # Analyze files
    result_file = analyze_demo_files(demo_folder)
    
    if result_file:
        print(f"\n🎉 Demo completed successfully!")
        print(f"Check the results in: {result_file}")
        print(f"\nTo analyze your own WAV files, run: uv run python main.py")
    
if __name__ == "__main__":
    main()