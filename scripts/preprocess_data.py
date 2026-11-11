"""Clean the sessions, build the hourly demand series, run EDA and train the demand forecaster."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_loader, demand_forecasting, eda, preprocessing


def main():
    report = preprocessing.run_pipeline()
    print("Cleaning report:", json.dumps(report, indent=2))
    stats = eda.run_eda()
    print(f"EDA done: {stats['n_sessions']} sessions, peak hour {stats['peak_hour_of_day']}:00 "
          f"[source: {data_loader.get_data_source()}]")
    res = demand_forecasting.train_and_evaluate()
    for name, m in res.metrics["metrics"].items():
        print(f"  {name:32s} MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.3f}")


if __name__ == "__main__":
    main()
