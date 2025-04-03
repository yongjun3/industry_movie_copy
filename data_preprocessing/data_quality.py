import os
import re
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import scipy.stats

os.makedirs("log", exist_ok=True)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='log/data_quality.log'
)
logger = logging.getLogger('data_quality')

class DataQualityMonitor:
    def __init__(self):
        self.baseline_stats = None
        self.drift_threshold = 0.3
        self.schema_violations = 0
        self.total_records = 0
        
    def log_error(self, message):
        """Log error messages"""
        logger.error(message)
        
    def log_warning(self, message):
        """Log warning messages"""
        logger.warning(message)
        
    def log_info(self, message):
        """Log info messages"""
        logger.info(message)
        
    def validate_log_entry(self, log_entry):
        """Validates that a log entry conforms to expected schema.
        
        Args:
            log_entry (str): The log entry to validate
            
        Returns:
            bool: True if the entry is valid, False otherwise
        """
        self.total_records += 1
        
        # Less than one element or not a string from kafka
        if not isinstance(log_entry, str) or len(log_entry.strip()) == 0:
            self.log_error("Invalid log entry format: empty or non-string")
            self.schema_violations += 1
            return False
        
        # Check if the log entry has the expected format (timestamp, user_id, request)
        parts = log_entry.strip().split(",")
        if len(parts) < 3:
            self.log_error(f"Schema violation: log entry missing required parts: {log_entry}")
            self.schema_violations += 1
            return False
        
        # Expected patterns check for our application
        expected_patterns = [
            re.compile(r'GET /rate/(\d+)=(\d)'),  # Rating pattern
            re.compile(r'GET /data/m/([^/]+)/(\d+)\.mpg')  # Watch pattern
        ]
        
        request_part = parts[2]
        if not any(pattern.search(request_part) for pattern in expected_patterns):
            self.log_error(f"Schema violation: request doesn't match expected patterns: {request_part}")
            self.schema_violations += 1
            return False
        
        return True
    
    def validate_api_response(self, response, expected_fields=None):
        """Validates that an API response has the expected structure
        
        Args:
            response (dict): The API response to validate
            expected_fields (list): List of expected fields in the response
            
        Returns:
            bool: True if the response is valid, False otherwise
        """
        if not expected_fields:
            return True
            
        if not isinstance(response, dict):
            self.log_error(f"API response is not a dictionary: {response}")
            return False
            
        missing_fields = [field for field in expected_fields if field not in response]
        if missing_fields:
            self.log_error(f"API response missing expected fields: {missing_fields}")
            return False
            
        return True
    
    def calculate_js_divergence(self, dist1, dist2):
        """Calculate Jensen-Shannon divergence between two distributions
        
        Args:
            dist1 (pd.Series): First distribution
            dist2 (pd.Series): Second distribution
            
        Returns:
            float: JS divergence value
        """
        # Ensure both distributions have the same keys
        all_keys = set(dist1.index).union(set(dist2.index))
        
        # Create normalized distributions with the same keys
        p = pd.Series({k: dist1.get(k, 0) for k in all_keys})
        q = pd.Series({k: dist2.get(k, 0) for k in all_keys})
        
        # Normalize if not already normalized
        if abs(p.sum() - 1.0) > 1e-10:
            p = p / p.sum()
        if abs(q.sum() - 1.0) > 1e-10:
            q = q / q.sum()
            
        # Calculate the mean distribution
        m = (p + q) / 2
        
        # Calculate JS divergence
        js_div = 0.5 * (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m))
        
        # Handle potential numerical issues
        if np.isnan(js_div):
            return 0.0
            
        return js_div
    
    def calculate_genre_distribution(self, df):
        """Calculate the distribution of genres in the data
        
        Args:
            df (pd.DataFrame): DataFrame containing movie data
            
        Returns:
            pd.Series: Distribution of genres
        """
        theme_cols = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
                      "Drama", "Family", "Fantasy", "Foreign", "History", "Horror", "Music",
                      "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"]
        
        # Ensure at least some genre columns exist
        available_cols = [col for col in theme_cols if col in df.columns]
        
        if not available_cols:
            self.log_warning("No genre columns found in DataFrame")
            return pd.Series()
        
        # Sum the occurrences of each genre and normalize
        genre_counts = df[available_cols].sum()
        return genre_counts / genre_counts.sum() if genre_counts.sum() > 0 else genre_counts
    
    def establish_baseline(self, df):
        """Establish baseline statistics for drift detection
        
        Args:
            df (pd.DataFrame): DataFrame with historical data
            
        Returns:
            dict: Baseline statistics
        """
        self.log_info("Establishing baseline statistics for drift detection")
        
        baseline = {
            'timestamp': datetime.now().isoformat(),
            'record_count': len(df),
            'rating_dist': None,
            'genre_dist': None,
            'avg_user_activity': None,
            'user_count': None,
            'movie_count': None
        }
        
        # Rating distribution
        if 'rating' in df.columns:
            baseline['rating_dist'] = df['rating'].value_counts(normalize=True)
            self.log_info(f"Baseline rating distribution: {baseline['rating_dist'].to_dict()}")
        
        # Genre distribution
        baseline['genre_dist'] = self.calculate_genre_distribution(df)
        if not baseline['genre_dist'].empty:
            self.log_info(f"Baseline genre distribution established with {len(baseline['genre_dist'])} genres")
        
        # User activity patterns
        if 'user_id' in df.columns:
            user_activity = df.groupby('user_id').size()
            baseline['avg_user_activity'] = user_activity.mean()
            baseline['user_activity_std'] = user_activity.std()
            baseline['user_count'] = df['user_id'].nunique()
            self.log_info(f"Baseline user activity: avg={baseline['avg_user_activity']:.2f}, users={baseline['user_count']}")
        
        # Movie statistics
        if 'movie_id' in df.columns:
            baseline['movie_count'] = df['movie_id'].nunique()
            self.log_info(f"Baseline movie count: {baseline['movie_count']}")
        
        self.baseline_stats = baseline
        self.log_info(f"Baseline established with {len(df)} records")
        
        return baseline
    
    def detect_data_drift(self, current_data):
        """Detects significant drift in data distributions compared to baseline
        
        Args:
            current_data (pd.DataFrame): Current data sample to check for drift
            
        Returns:
            dict: Drift detection results
        """
        if self.baseline_stats is None:
            self.log_warning("No baseline statistics available for drift detection")
            return {'drift_detected': False, 'message': 'No baseline available'}
        
        result = {
            'drift_detected': False,
            'drift_metrics': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate current rating distribution
        if 'rating' in current_data.columns and self.baseline_stats['rating_dist'] is not None:
            current_rating_dist = current_data['rating'].value_counts(normalize=True)
            js_divergence = self.calculate_js_divergence(current_rating_dist, self.baseline_stats['rating_dist'])
            result['drift_metrics']['rating_js_divergence'] = js_divergence
            
            if js_divergence > self.drift_threshold:
                result['drift_detected'] = True
                result['drift_metrics']['rating_drift'] = True
                self.log_warning(f"Rating distribution drift detected: JS div={js_divergence:.4f}")
        
        # Check for drift in genre distribution
        current_genre_dist = self.calculate_genre_distribution(current_data)
        if not current_genre_dist.empty and not self.baseline_stats['genre_dist'].empty:
            genre_js_div = self.calculate_js_divergence(current_genre_dist, self.baseline_stats['genre_dist'])
            result['drift_metrics']['genre_js_divergence'] = genre_js_div
            
            if genre_js_div > self.drift_threshold:
                result['drift_detected'] = True
                result['drift_metrics']['genre_drift'] = True
                self.log_warning(f"Genre distribution drift detected: JS div={genre_js_div:.4f}")
        
        # Check for drift in user activity patterns
        if 'user_id' in current_data.columns and self.baseline_stats['avg_user_activity'] is not None:
            current_user_activity = current_data.groupby('user_id').size()
            current_avg_activity = current_user_activity.mean()
            
            activity_change_ratio = abs(current_avg_activity - self.baseline_stats['avg_user_activity']) / self.baseline_stats['avg_user_activity']
            result['drift_metrics']['user_activity_change_ratio'] = activity_change_ratio
            
            if activity_change_ratio > self.drift_threshold:
                result['drift_detected'] = True
                result['drift_metrics']['user_activity_drift'] = True
                self.log_warning(f"User activity drift detected: change ratio={activity_change_ratio:.4f}")
        
        # Check new vs. baseline record counts
        if len(current_data) > 0 and self.baseline_stats['record_count'] > 0:
            size_ratio = len(current_data) / self.baseline_stats['record_count']
            result['drift_metrics']['size_ratio'] = size_ratio
            
            # Size imbalance can indicate potential sampling issues
            if size_ratio < 0.1 or size_ratio > 10:
                self.log_warning(f"Sample size imbalance detected: current={len(current_data)}, baseline={self.baseline_stats['record_count']}")
        
        # Log and return result
        if result['drift_detected']:
            self.log_warning(f"Data drift detected! Metrics: {json.dumps(result['drift_metrics'])}")
        
        return result
    
    def get_schema_quality_metrics(self):
        """Get current schema validation metrics
        
        Returns:
            dict: Schema quality metrics
        """
        violation_rate = self.schema_violations / self.total_records if self.total_records > 0 else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'schema_violations': self.schema_violations,
            'total_records': self.total_records,
            'violation_rate': violation_rate
        }
    
    def get_drift_quality_metrics(self):
        """Get current drift detection metrics
        
        Returns:
            dict: Drift quality metrics
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'drift_threshold': self.drift_threshold,
            'baseline_established': self.baseline_stats is not None,
            'baseline_timestamp': self.baseline_stats['timestamp'] if self.baseline_stats else None
        }
    
    def get_data_quality_metrics(self):
        """Get all data quality metrics
        
        Returns:
            dict: Combined data quality metrics
        """
        schema_metrics = self.get_schema_quality_metrics()
        drift_metrics = self.get_drift_quality_metrics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'schema_metrics': schema_metrics,
            'drift_metrics': drift_metrics
        }

# Global instance for use throughout the application
data_quality_monitor = DataQualityMonitor()
