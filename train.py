import os
import numpy as np
import librosa
import joblib
import warnings
import logging
import argparse
import hashlib
import yaml
from datetime import datetime
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from multiprocessing import cpu_count

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
def setup_logging(debug=False):
    """Configure logging system"""
    log_level = logging.DEBUG if debug else logging.INFO
    log_filename = f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Configuration class
class Config:
    """Configuration management"""
    def __init__(self, config_file=None):
        # Default configuration
        self.data_dir = "Data"
        self.sample_rate = 22050
        self.duration = 5
        self.n_mfcc = 13
        self.model_path = "voice_classifier.pkl"
        self.cache_dir = ".feature_cache"
        self.n_cores = cpu_count()
        self.test_size = 0.15
        self.val_size = 0.2
        self.random_state = 42
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file):
        """Load configuration from YAML file"""
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
            for key, value in config_data.items():
                setattr(self, key, value)
    
    def save_to_file(self, config_file):
        """Save configuration to YAML file"""
        config_data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)

def get_cache_key(file_path):
    """Generate cache key from file metadata"""
    try:
        stat = os.stat(file_path)
        key = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(key.encode()).hexdigest()
    except Exception:
        return None

def augment_audio(y, sr):
    """Create augmented versions of audio for training"""
    augmented = [y]
    
    try:
        # Time stretch (slightly faster)
        y_stretch = librosa.effects.time_stretch(y, rate=1.1)
        augmented.append(y_stretch)
        
        # Pitch shift (up by 2 semitones)
        y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        augmented.append(y_pitch)
        
        # Add slight noise
        noise = np.random.randn(len(y)) * 0.003
        y_noise = y + noise
        augmented.append(y_noise)
    except Exception as e:
        logger.warning(f"Augmentation failed: {e}")
    
    return augmented

def extract_features(file_path, config, use_augmentation=False):
    """Extract comprehensive audio features"""
    try:
        # Validate file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"Empty file: {file_path}")
        
        # Load audio file
        y, sr = librosa.load(file_path, sr=config.sample_rate, duration=config.duration)
        
        if len(y) == 0:
            logger.warning(f"Empty audio data: {file_path}")
            return None
        
        # Pad if too short
        if len(y) < config.duration * config.sample_rate:
            y = np.pad(y, (0, config.duration * config.sample_rate - len(y)), 'constant')
        
        # Extract features from audio
        features = extract_single_features(y, sr, config)
        
        # Return augmented features if requested
        if use_augmentation:
            all_features = [features]
            augmented_audios = augment_audio(y, sr)
            for aug_y in augmented_audios[1:]:  # Skip original
                aug_features = extract_single_features(aug_y, sr, config)
                if aug_features is not None:
                    all_features.append(aug_features)
            return all_features
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features from {file_path}: {e}")
        return None

def extract_single_features(y, sr, config):
    """Extract features from a single audio signal"""
    features = []
    
    try:
        # 1. MFCCs (mean and std)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        features.extend(mfccs_mean)
        features.extend(mfccs_std)
        
        # 2. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.append(np.mean(spectral_centroid))
        features.append(np.std(spectral_centroid))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features.append(np.mean(spectral_rolloff))
        features.append(np.std(spectral_rolloff))
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features.append(np.mean(spectral_bandwidth))
        features.append(np.std(spectral_bandwidth))
        
        # 3. Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.append(np.mean(chroma))
        features.append(np.std(chroma))
        
        # 4. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        
        # 5. Energy features
        features.append(np.mean(np.abs(y)))
        features.append(np.std(y))
        features.append(np.max(np.abs(y)))
        
        # 6. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo)
        
        return np.array(features)
        
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return None

def get_feature_names(config):
    """Generate feature names for interpretability"""
    names = []
    
    # MFCCs mean
    for i in range(config.n_mfcc):
        names.append(f"MFCC_{i+1}_mean")
    
    # MFCCs std
    for i in range(config.n_mfcc):
        names.append(f"MFCC_{i+1}_std")
    
    # Spectral features
    names.extend([
        "Spectral_Centroid_mean", "Spectral_Centroid_std",
        "Spectral_Rolloff_mean", "Spectral_Rolloff_std",
        "Spectral_Bandwidth_mean", "Spectral_Bandwidth_std",
        "Chroma_mean", "Chroma_std",
        "ZCR_mean", "ZCR_std",
        "Energy_mean", "Energy_std", "Energy_max",
        "Tempo"
    ])
    
    return names

def load_data(config, limit_per_class=None, use_cache=True, use_augmentation=False):
    """Load audio files and extract features with caching"""
    X = []
    y = []
    class_indices = {}
    file_paths = []
    
    logger.info("Loading and extracting features from audio files...")
    
    # Create cache directory
    if use_cache:
        os.makedirs(config.cache_dir, exist_ok=True)
    
    # Get all class folders
    class_folders = [f for f in os.listdir(config.data_dir) 
                     if os.path.isdir(os.path.join(config.data_dir, f))]
    
    for class_idx, class_name in enumerate(sorted(class_folders)):
        class_indices[class_idx] = class_name
        class_dir = os.path.join(config.data_dir, class_name)
        logger.info(f"Processing class: {class_name} ({class_idx})")
        
        # Get all audio files
        audio_files = [f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac', '.m4a'))]
        logger.info(f"Found {len(audio_files)} audio files in {class_name}")
        
        # Limit files if specified
        if limit_per_class and len(audio_files) > limit_per_class:
            logger.info(f"Limiting to {limit_per_class} files per class")
            audio_files = sorted(audio_files)[:limit_per_class]
        
        success_count = 0
        for audio_file in tqdm(audio_files, desc=f"Processing {class_name}"):
            file_path = os.path.join(class_dir, audio_file)
            
            # Try to load from cache
            features_list = None
            if use_cache:
                cache_key = get_cache_key(file_path)
                if cache_key:
                    cache_path = os.path.join(config.cache_dir, f"{cache_key}.npy")
                    if os.path.exists(cache_path):
                        try:
                            features_list = [np.load(cache_path)]
                        except Exception:
                            pass
            
            # Extract features if not cached
            if features_list is None:
                result = extract_features(file_path, config, use_augmentation)
                
                if result is None:
                    continue
                
                # Handle both single and augmented features
                if use_augmentation and isinstance(result, list):
                    features_list = result
                else:
                    features_list = [result]
                
                # Cache the original features
                if use_cache and features_list:
                    cache_key = get_cache_key(file_path)
                    if cache_key:
                        cache_path = os.path.join(config.cache_dir, f"{cache_key}.npy")
                        try:
                            np.save(cache_path, features_list[0])
                        except Exception as e:
                            logger.warning(f"Failed to cache features: {e}")
            
            # Add all features to dataset
            for features in features_list:
                if features is not None:
                    X.append(features)
                    y.append(class_idx)
                    file_paths.append(file_path)
                    success_count += 1
        
        logger.info(f"Successfully processed {success_count} samples for class {class_name}")
    
    if len(X) == 0:
        logger.error("No features could be extracted. Check your audio files.")
        return np.array([]), np.array([]), {}, []
    
    logger.info(f"Total extracted features: {len(X)} from {len(class_folders)} classes")
    logger.info(f"Class distribution: {dict(Counter(y))}")
    
    return np.array(X), np.array(y), class_indices, file_paths

def find_optimal_model(X_train, y_train, X_val, y_val, config):
    """Find optimal model through grid search"""
    logger.info("\nFinding optimal model parameters...")
    
    # 1. Random Forest optimization
    logger.info("Optimizing Random Forest...")
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=config.random_state, 
                               n_jobs=config.n_cores, 
                               class_weight='balanced'),
        rf_params,
        cv=3,
        n_jobs=config.n_cores,
        verbose=1,
        scoring='accuracy'
    )
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_
    rf_score = accuracy_score(y_val, rf_best.predict(X_val))
    logger.info(f"RF best params: {rf_grid.best_params_}")
    logger.info(f"RF validation accuracy: {rf_score:.4f}")
    
    # Cross-validation score
    rf_cv_scores = cross_val_score(rf_best, X_train, y_train, cv=5, n_jobs=config.n_cores)
    logger.info(f"RF CV Accuracy: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std() * 2:.4f})")
    
    # 2. Gradient Boosting optimization
    logger.info("\nOptimizing Gradient Boosting...")
    gb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
    
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=config.random_state),
        gb_params,
        cv=3,
        n_jobs=config.n_cores,
        verbose=1,
        scoring='accuracy'
    )
    gb_grid.fit(X_train, y_train)
    gb_best = gb_grid.best_estimator_
    gb_score = accuracy_score(y_val, gb_best.predict(X_val))
    logger.info(f"GB best params: {gb_grid.best_params_}")
    logger.info(f"GB validation accuracy: {gb_score:.4f}")
    
    # 3. SVM optimization (only for smaller datasets)
    svm_best = None
    svm_score = 0
    
    if len(X_train) < 5000:
        logger.info("\nOptimizing SVM...")
        svm_params = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf']
        }
        
        svm_grid = GridSearchCV(
            SVC(probability=True, random_state=config.random_state),
            svm_params,
            cv=3,
            n_jobs=config.n_cores,
            verbose=1,
            scoring='accuracy'
        )
        svm_grid.fit(X_train, y_train)
        svm_best = svm_grid.best_estimator_
        svm_score = accuracy_score(y_val, svm_best.predict(X_val))
        logger.info(f"SVM best params: {svm_grid.best_params_}")
        logger.info(f"SVM validation accuracy: {svm_score:.4f}")
    else:
        logger.info("Skipping SVM due to large dataset size")
    
    # Create weighted voting ensemble
    models = []
    weights = []
    
    models.append(('rf', rf_best))
    weights.append(rf_score)
    
    models.append(('gb', gb_best))
    weights.append(gb_score)
    
    if svm_best is not None:
        models.append(('svm', svm_best))
        weights.append(svm_score)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    logger.info(f"\nEnsemble weights: {dict(zip([m[0] for m in models], weights))}")
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=models,
        voting='soft',
        weights=weights
    )
    
    # Train ensemble
    ensemble.fit(X_train, y_train)
    
    return {
        'ensemble': ensemble,
        'rf': rf_best,
        'gb': gb_best,
        'svm': svm_best,
        'weights': weights.tolist()
    }

def train_model(config, limit_per_class=None, use_augmentation=False):
    """Train an enhanced audio classifier"""
    logger.info("="*50)
    logger.info("AUDIO CLASSIFIER TRAINING")
    logger.info("="*50)
    
    # Load and extract features
    X, y, class_indices, file_paths = load_data(config, limit_per_class, 
                                                 use_augmentation=use_augmentation)
    
    if len(X) == 0:
        logger.error("No features extracted. Cannot train model.")
        return None
    
    # Standardize features
    logger.info("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y, test_size=config.test_size, 
        random_state=config.random_state, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=config.val_size, 
        random_state=config.random_state, stratify=y_train_val
    )
    
    logger.info(f"Training: {len(X_train)} samples")
    logger.info(f"Validation: {len(X_val)} samples")
    logger.info(f"Testing: {len(X_test)} samples")
    
    # Find optimal model
    models = find_optimal_model(X_train, y_train, X_val, y_val, config)
    
    # Train on combined train+val for final evaluation
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    logger.info("\nTraining final model on full training set...")
    models['ensemble'].fit(X_train_full, y_train_full)
    
    # Evaluate on test set
    y_pred = models['ensemble'].predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info("\n" + "="*50)
    logger.info("FINAL MODEL PERFORMANCE")
    logger.info("="*50)
    logger.info(f"Ensemble model accuracy: {accuracy:.4f}")
    
    class_names = [class_indices[i] for i in sorted(class_indices.keys())]
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    for i, class_name in enumerate(class_names):
        logger.info(f"{class_name}: {i}")
    logger.info("\n" + str(cm))
    
    # Feature importance
    feature_names = get_feature_names(config)
    importances = models['rf'].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    logger.info("\nTop 15 Most Important Features:")
    for i in range(min(15, len(feature_names))):
        idx = indices[i]
        logger.info(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Save model
    logger.info(f"\nSaving model to {config.model_path}...")
    model_data = {
        'ensemble': models['ensemble'],
        'rf': models['rf'],
        'gb': models['gb'],
        'svm': models['svm'],
        'scaler': scaler,
        'class_indices': class_indices,
        'feature_names': feature_names,
        'weights': models['weights'],
        'accuracy': accuracy,
        'config': config,
        'training_date': datetime.now().isoformat(),
        'training_samples': len(X_train_full),
        'test_samples': len(X_test),
        'class_distribution': dict(Counter(y))
    }
    
    joblib.dump(model_data, config.model_path)
    logger.info(f"Model successfully saved!")
    
    return model_data

def predict(audio_file, model_data=None, model_path=None):
    """Make prediction on a new audio file"""
    if model_data is None:
        if model_path is None:
            model_path = "voice_classifier.pkl"
        
        try:
            model_data = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Could not load model from {model_path}: {e}")
            return None
    
    # Get config
    config = model_data.get('config', Config())
    
    # Extract features
    features = extract_features(audio_file, config, use_augmentation=False)
    
    if features is None:
        logger.error("Could not extract features from audio file")
        return None
    
    # Scale features
    features_scaled = model_data['scaler'].transform([features])
    
    # Get models
    ensemble = model_data['ensemble']
    rf = model_data['rf']
    gb = model_data['gb']
    svm = model_data.get('svm', None)
    
    # Ensemble prediction
    ensemble_prediction = ensemble.predict(features_scaled)[0]
    ensemble_proba = ensemble.predict_proba(features_scaled)[0]
    
    # Individual predictions
    individual_predictions = {}
    
    rf_prediction = rf.predict(features_scaled)[0]
    rf_proba = rf.predict_proba(features_scaled)[0]
    individual_predictions["Random Forest"] = {
        "class": model_data['class_indices'][rf_prediction],
        "confidence": float(rf_proba[rf_prediction])
    }
    
    gb_prediction = gb.predict(features_scaled)[0]
    gb_proba = gb.predict_proba(features_scaled)[0]
    individual_predictions["Gradient Boosting"] = {
        "class": model_data['class_indices'][gb_prediction],
        "confidence": float(gb_proba[gb_prediction])
    }
    
    if svm is not None:
        svm_prediction = svm.predict(features_scaled)[0]
        svm_proba = svm.predict_proba(features_scaled)[0]
        individual_predictions["SVM"] = {
            "class": model_data['class_indices'][svm_prediction],
            "confidence": float(svm_proba[svm_prediction])
        }
    
    # Result
    class_indices = model_data['class_indices']
    class_name = class_indices[ensemble_prediction]
    
    result = {
        'class': class_name,
        'confidence': float(ensemble_proba[ensemble_prediction]),
        'probabilities': {class_indices[i]: float(prob) for i, prob in enumerate(ensemble_proba)},
        'individual_models': individual_predictions,
        'model_weights': model_data.get('weights', None)
    }
    
    return result

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Enhanced Audio Classifier')
    parser.add_argument('--config', type=str, help='Path to config file (YAML)')
    parser.add_argument('--data-dir', type=str, default='Data', help='Directory with audio files')
    parser.add_argument('--model-path', type=str, default='voice_classifier.pkl', help='Path to save/load model')
    parser.add_argument('--limit', type=int, help='Limit samples per class')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--predict', type=str, help='Audio file to classify')
    parser.add_argument('--no-cache', action='store_true', help='Disable feature caching')
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(args.debug)
    
    # Load configuration
    config = Config(args.config)
    
    # Override with command line arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.model_path:
        config.model_path = args.model_path
    
    # Prediction mode
    if args.predict:
        logger.info(f"Predicting class for: {args.predict}")
        result = predict(args.predict, model_path=config.model_path)
        
        if result:
            logger.info("\n" + "="*50)
            logger.info("PREDICTION RESULT")
            logger.info("="*50)
            logger.info(f"Predicted Class: {result['class']}")
            logger.info(f"Confidence: {result['confidence']*100:.2f}%")
            logger.info("\nAll Probabilities:")
            for cls, prob in result['probabilities'].items():
                logger.info(f"  {cls}: {prob*100:.2f}%")
            logger.info("\nIndividual Model Predictions:")
            for model_name, pred in result['individual_models'].items():
                logger.info(f"  {model_name}: {pred['class']} ({pred['confidence']*100:.2f}%)")
        return
    
    # Training mode
    logger.info("="*50)
    logger.info("ENHANCED AUDIO CLASSIFIER")
    logger.info("="*50)
    
    # Validate data directory
    if not os.path.exists(config.data_dir):
        logger.error(f"Error: {config.data_dir} directory not found!")
        return
    
    subdirs = [d for d in os.listdir(config.data_dir) 
               if os.path.isdir(os.path.join(config.data_dir, d))]
    
    if len(subdirs) < 2:
        logger.error(f"Found only {len(subdirs)} classes. Need at least 2 classes.")
        return
    
    logger.info(f"Found {len(subdirs)} classes: {', '.join(subdirs)}")
    logger.info(f"Using {config.n_cores} CPU cores")
    logger.info(f"Data augmentation: {'enabled' if args.augment else 'disabled'}")
    logger.info(f"Feature caching: {'disabled' if args.no_cache else 'enabled'}")
    
    # Train model
    train_model(config, limit_per_class=args.limit, use_augmentation=args.augment)
    
    logger.info("\nTraining complete!")

if __name__ == "__main__":
    main()
