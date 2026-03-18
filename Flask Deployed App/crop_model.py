import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_crop_data():
    # Load the fertilizer data from the correct path
    return pd.read_csv('fertilizer.csv')

def calculate_crop_suitability(n, p, k):
    # Load and prepare data
    try:
        df = load_crop_data()
    except Exception as e:
        print(f"Error loading crop data: {str(e)}")
        return []
    
    # Create feature matrix from input parameters
    input_features = np.array([n, p, k])
    
    # Calculate similarity scores
    scores = []
    for _, row in df.iterrows():
        crop_features = np.array([row['N'], row['P'], row['K']])
        # Calculate Euclidean distance (lower is better)
        distance = np.sqrt(np.sum((input_features - crop_features) ** 2))
        scores.append({
            'crop': row['Crop'],
            'requirements': {
                'N': row['N'],
                'P': row['P'],
                'K': row['K']
            }
        })
    
    # Sort crops by distance (lower is better)
    sorted_crops = sorted(scores, key=lambda x: np.sqrt(np.sum((input_features - np.array([x['requirements']['N'], x['requirements']['P'], x['requirements']['K']])) ** 2)))
    
    # Return top 3 recommendations
    return sorted_crops[:3]

def get_recommendation_text(recommendations, input_params):
    result = "Based on the soil nutrient parameters:\n\n"
    result += f"Top Recommended Crops:\n\n"
    
    for i, rec in enumerate(recommendations, 1):
        result += f"{i}. {rec['crop'].title()}\n"
        result += f"   Optimal NPK values: N={rec['requirements']['N']}, "
        result += f"P={rec['requirements']['P']}, K={rec['requirements']['K']}\n\n"
    
    result += f"\nYour Soil Parameters:\n"
    result += f"Nitrogen (N): {input_params['N']}\n"
    result += f"Phosphorous (P): {input_params['P']}\n"
    result += f"Potassium (K): {input_params['K']}\n"
    
    return result