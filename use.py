"""
Audio Classifier Usage Script
Easy-to-use interface for predicting audio classifications
"""

import os
import sys
import argparse
import joblib
import numpy as np
import librosa
from pathlib import Path
from datetime import datetime

class AudioClassifier:
    """Simple interface for audio classification"""
    
    def __init__(self, model_path="voice_classifier.pkl"):
        """Initialize classifier with trained model"""
        self.model_path = model_path
        self.model_data = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                "Please train the model first using the main script."
            )
        
        try:
            self.model_data = joblib.load(self.model_path)
            print(f"✓ Model loaded successfully from {self.model_path}")
            self.print_model_info()
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
    
    def print_model_info(self):
        """Print model information"""
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        
        if 'training_date' in self.model_data:
            print(f"Training Date: {self.model_data['training_date']}")
        
        if 'accuracy' in self.model_data:
            print(f"Model Accuracy: {self.model_data['accuracy']*100:.2f}%")
        
        if 'class_indices' in self.model_data:
            classes = sorted(self.model_data['class_indices'].items())
            print(f"\nAvailable Classes ({len(classes)}):")
            for idx, name in classes:
                print(f"  {idx}: {name}")
        
        if 'training_samples' in self.model_data:
            print(f"\nTraining Samples: {self.model_data['training_samples']}")
        
        if 'class_distribution' in self.model_data:
            print("\nClass Distribution:")
            for cls, count in self.model_data['class_distribution'].items():
                class_name = self.model_data['class_indices'].get(cls, f"Class {cls}")
                print(f"  {class_name}: {count} samples")
        
        print("="*60 + "\n")
    
    def extract_features(self, file_path):
        """Extract features from audio file"""
        try:
            config = self.model_data.get('config')
            if config is None:
                # Use default values if config not saved
                sample_rate = 22050
                duration = 5
                n_mfcc = 13
            else:
                sample_rate = config.sample_rate
                duration = config.duration
                n_mfcc = config.n_mfcc
            
            # Load audio
            y, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
            
            if len(y) == 0:
                raise ValueError("Empty audio file")
            
            # Pad if too short
            if len(y) < duration * sample_rate:
                y = np.pad(y, (0, duration * sample_rate - len(y)), 'constant')
            
            # Extract features
            features = []
            
            # MFCCs (mean and std)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            mfccs_std = np.std(mfccs.T, axis=0)
            features.extend(mfccs_mean)
            features.extend(mfccs_std)
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features.append(np.mean(spectral_centroid))
            features.append(np.std(spectral_centroid))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features.append(np.mean(spectral_rolloff))
            features.append(np.std(spectral_rolloff))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features.append(np.mean(spectral_bandwidth))
            features.append(np.std(spectral_bandwidth))
            
            # Chroma
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.append(np.mean(chroma))
            features.append(np.std(chroma))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.append(np.mean(zcr))
            features.append(np.std(zcr))
            
            # Energy
            features.append(np.mean(np.abs(y)))
            features.append(np.std(y))
            features.append(np.max(np.abs(y)))
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.append(tempo)
            
            return np.array(features)
            
        except Exception as e:
            raise Exception(f"Error extracting features: {e}")
    
    def predict(self, audio_file, verbose=True):
        """Predict class for audio file"""
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        if verbose:
            print(f"\nAnalyzing: {audio_file}")
            print("-" * 60)
        
        # Extract features
        features = self.extract_features(audio_file)
        
        # Scale features
        features_scaled = self.model_data['scaler'].transform([features])
        
        # Get models
        ensemble = self.model_data['ensemble']
        
        # Make prediction
        prediction = ensemble.predict(features_scaled)[0]
        probabilities = ensemble.predict_proba(features_scaled)[0]
        
        # Get class name
        class_indices = self.model_data['class_indices']
        predicted_class = class_indices[prediction]
        confidence = probabilities[prediction]
        
        # Create result
        result = {
            'file': audio_file,
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {class_indices[i]: float(prob) 
                            for i, prob in enumerate(probabilities)},
            'timestamp': datetime.now().isoformat()
        }
        
        if verbose:
            self.print_result(result)
        
        return result
    
    def print_result(self, result):
        """Print prediction result in a nice format"""
        print(f"\n{'='*60}")
        print("PREDICTION RESULT")
        print(f"{'='*60}")
        print(f"\nPredicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        
        # Confidence bar
        bar_length = 40
        filled = int(bar_length * result['confidence'])
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"   [{bar}]")
        
        print("\nAll Class Probabilities:")
        # Sort by probability (descending)
        sorted_probs = sorted(result['probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)
        
        for class_name, prob in sorted_probs:
            bar_length = 30
            filled = int(bar_length * prob)
            bar = '█' * filled + '░' * (bar_length - filled)
            marker = "👉 " if class_name == result['predicted_class'] else "   "
            print(f"{marker}{class_name:20s} [{bar}] {prob*100:6.2f}%")
        
        print(f"\n{'='*60}\n")
    
    def predict_batch(self, audio_files, output_file=None):
        """Predict classes for multiple audio files"""
        print(f"\nBatch Processing: {len(audio_files)} files")
        print("="*60)
        
        results = []
        successful = 0
        failed = 0
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file}")
            
            try:
                result = self.predict(audio_file, verbose=False)
                results.append(result)
                successful += 1
                
                # Print compact result
                print(f"  ✓ {result['predicted_class']} ({result['confidence']*100:.1f}%)")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                failed += 1
                results.append({
                    'file': audio_file,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Summary
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total files: {len(audio_files)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        # Save results if output file specified
        if output_file:
            self.save_results(results, output_file)
        
        return results
    
    def save_results(self, results, output_file):
        """Save results to file"""
        import json
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            print(f"\nError saving results: {e}")
    
    def predict_directory(self, directory, recursive=False, output_file=None):
        """Predict all audio files in a directory"""
        audio_extensions = {'.wav', '.mp3', '.ogg', '.flac', '.m4a'}
        
        if recursive:
            audio_files = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if Path(file).suffix.lower() in audio_extensions:
                        audio_files.append(os.path.join(root, file))
        else:
            audio_files = [
                os.path.join(directory, f) 
                for f in os.listdir(directory)
                if Path(f).suffix.lower() in audio_extensions
            ]
        
        if not audio_files:
            print(f"No audio files found in {directory}")
            return []
        
        print(f"Found {len(audio_files)} audio files in {directory}")
        return self.predict_batch(audio_files, output_file)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Audio Classifier - Predict audio classifications',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict single file
  python use.py audio.wav
  
  # Predict multiple files
  python use.py audio1.wav audio2.wav audio3.wav
  
  # Predict all files in directory
  python use.py --directory ./test_audio/
  
  # Predict recursively with output file
  python use.py --directory ./test_audio/ --recursive --output results.json
  
  # Use custom model
  python use.py audio.wav --model my_model.pkl
        """
    )
    
    parser.add_argument('files', nargs='*', help='Audio files to classify')
    parser.add_argument('--model', '-m', default='voice_classifier.pkl',
                       help='Path to trained model (default: voice_classifier.pkl)')
    parser.add_argument('--directory', '-d', help='Directory containing audio files')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Process directories recursively')
    parser.add_argument('--output', '-o', help='Save results to JSON file')
    parser.add_argument('--info', action='store_true',
                       help='Show model information and exit')
    
    args = parser.parse_args()
    
    # Initialize classifier
    try:
        classifier = AudioClassifier(model_path=args.model)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Show info and exit
    if args.info:
        sys.exit(0)
    
    # Process directory
    if args.directory:
        if not os.path.isdir(args.directory):
            print(f"Error: Directory not found: {args.directory}")
            sys.exit(1)
        
        classifier.predict_directory(args.directory, args.recursive, args.output)
        sys.exit(0)
    
    # Process individual files
    if not args.files:
        print("Error: No audio files specified")
        print("Use --help for usage information")
        sys.exit(1)
    
    # Single file
    if len(args.files) == 1:
        try:
            classifier.predict(args.files[0])
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Multiple files
        classifier.predict_batch(args.files, args.output)

if __name__ == "__main__":
    main()

