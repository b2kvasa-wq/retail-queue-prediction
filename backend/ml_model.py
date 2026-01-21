import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

class MLQueuePredictor:
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = ['time_of_day', 'day_of_week', 'store_traffic', 'peak_hours', 'avg_service_time', 'checkout_counters']
        self.is_trained = False
    
    def preprocess_data(self, data):
        try:
            if isinstance(data, dict):
                features = []
                for feature in self.feature_names:
                    if feature not in data:
                        raise ValueError(f'Missing feature: {feature}')
                    value = data[feature]
                    try:
                        features.append(float(value))
                    except (ValueError, TypeError):
                        raise ValueError(f'Invalid data type for {feature}. Expected numeric value.')
                data = np.array(features).reshape(1, -1)
            else:
                data = np.array(data).reshape(1, -1)
            
            if data.shape[1] != len(self.feature_names):
                raise ValueError(f'Expected {len(self.feature_names)} features')
            
            normalized = self.scaler.fit_transform(data)
            return normalized
        except Exception as e:
            raise ValueError(f'Data preprocessing error: {str(e)}')
    
    def predict(self, input_data):
        try:
            preprocessed_data = self.preprocess_data(input_data)
            if not self.is_trained:
                prediction = self._generate_demo_prediction(input_data)
            else:
                prediction = self.model.predict(preprocessed_data)[0]
            return float(np.clip(prediction, 0, 100))
        except Exception as e:
            raise ValueError(f'Prediction error: {str(e)}')
    
    def _generate_demo_prediction(self, input_data):
        base_queue = 15
        time_impact = input_data.get('time_of_day', 12) * 0.5
        day_impact = input_data.get('day_of_week', 3) * 2
        traffic_impact = input_data.get('store_traffic', 50) * 0.3
        peak_impact = input_data.get('peak_hours', 0) * 10
        service_impact = input_data.get('avg_service_time', 2) * 3
        counter_impact = input_data.get('checkout_counters', 3) * (-5)
        prediction = base_queue + time_impact + day_impact + traffic_impact + peak_impact + service_impact + counter_impact
        return np.clip(prediction, 0, 100)
