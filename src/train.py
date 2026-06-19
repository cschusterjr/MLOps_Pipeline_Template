import pathlib
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


OUT = pathlib.Path("artifacts")
OUT.mkdir(exist_ok=True, parents=True)

DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

FEATURE_COLUMNS = [
    "attendance_rate",
    "quiz_average",
    "homework_average",
    "exam_average",
    "missing_assignments",
    "late_submissions",
    "engagement_score",
    "current_grade",
]


def generate_student_data(n_rows: int = 10000, random_state: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    attendance_rate = rng.normal(85, 10, n_rows).clip(40, 100)
    quiz_average = rng.normal(78, 12, n_rows).clip(30, 100)
    homework_average = rng.normal(80, 11, n_rows).clip(25, 100)
    exam_average = rng.normal(76, 14, n_rows).clip(20, 100)
    missing_assignments = rng.poisson(2, n_rows).clip(0, 12)
    late_submissions = rng.poisson(3, n_rows).clip(0, 15)
    engagement_score = rng.normal(70, 18, n_rows).clip(0, 100)

    current_grade = (
        quiz_average * 0.25
        + homework_average * 0.25
        + exam_average * 0.35
        + attendance_rate * 0.10
        + engagement_score * 0.05
        - missing_assignments * 1.5
        - late_submissions * 0.6
        + rng.normal(0, 4, n_rows)
    ).clip(0, 100)

    final_grade = (
        current_grade * 0.55
        + exam_average * 0.20
        + homework_average * 0.10
        + attendance_rate * 0.10
        + engagement_score * 0.05
        - missing_assignments * 1.2
        - late_submissions * 0.4
        + rng.normal(0, 3, n_rows)
    ).clip(0, 100)

    return pd.DataFrame(
        {
            "attendance_rate": attendance_rate.round(1),
            "quiz_average": quiz_average.round(1),
            "homework_average": homework_average.round(1),
            "exam_average": exam_average.round(1),
            "missing_assignments": missing_assignments,
            "late_submissions": late_submissions,
            "engagement_score": engagement_score.round(1),
            "current_grade": current_grade.round(1),
            "final_grade": final_grade.round(1),
        }
    )


def main():
    df = generate_student_data()
    data_path = DATA_DIR / "student_success_data.csv"
    df.to_csv(data_path, index=False)

    X = df[FEATURE_COLUMNS]
    y = df["final_grade"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7
    )

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=7,
        max_depth=10,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "mean_absolute_error": round(mean_absolute_error(y_test, preds), 3),
        "r2_score": round(r2_score(y_test, preds), 3),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "features": FEATURE_COLUMNS,
    }

    joblib.dump(model, OUT / "model.pkl")
    joblib.dump(FEATURE_COLUMNS, OUT / "feature_columns.pkl")

    pd.Series(metrics).to_json(OUT / "metrics.json", indent=2)

    print("Saved artifacts/model.pkl")
    print("Saved artifacts/feature_columns.pkl")
    print("Saved artifacts/metrics.json")
    print(f"Saved {data_path}")
    print(metrics)


if __name__ == "__main__":
    main()