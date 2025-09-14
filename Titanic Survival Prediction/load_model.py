import pickle
import joblib
import os

model_path = r"d:\My_Projects\Titanic Survival prediction\model.pkl"

if not os.path.exists(model_path):
    print("Model file not found!")
    exit()

# Try loading with pickle
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded with pickle.")
except Exception:
    # If pickle fails, try joblib
    model = joblib.load(model_path)
    print("Model loaded with joblib.")

print("Type of model:", type(model))
print(model)

