from src.data import load_and_clean_bike_data
from src.features import create_features
from src.train import train_and_evaluate

from pathlib import Path
from datetime import datetime

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)
Path("figures/results").mkdir(exist_ok=True)

df = load_and_clean_bike_data("data/hour.csv")
df = create_features(df)
results = train_and_evaluate(df)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results.to_csv(
    f"results/results_{timestamp}.csv",
    index=False,
)

print(results)