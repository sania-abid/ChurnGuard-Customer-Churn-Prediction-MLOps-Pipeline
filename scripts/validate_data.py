from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from churn_app.config import get_settings
from churn_app.data import ensure_dataset, load_training_frame


def _fallback_validate() -> bool:
    settings = get_settings()
    dataset_path = ensure_dataset(settings.raw_data_path, settings.dataset_url)
    frame = load_training_frame(dataset_path)

    checks = {
        "row_count": 5_000 <= len(frame) <= 8_000,
        "customer_id_not_null": frame["customerID"].notna().all(),
        "senior_citizen_binary": frame["SeniorCitizen"].isin([0, 1]).all(),
        "tenure_in_range": frame["tenure"].between(0, 72).all(),
        "monthly_charges_positive": frame["MonthlyCharges"].between(0, 200).all(),
        "total_charges_positive": frame["TotalCharges"].between(0, 10_000).all(),
        "churn_binary": frame["Churn"].isin(["Yes", "No"]).all(),
    }
    success = all(checks.values())
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    return success


def _great_expectations_validate() -> bool:
    import great_expectations as gx

    settings = get_settings()
    dataset_path = ensure_dataset(settings.raw_data_path, settings.dataset_url)
    frame = load_training_frame(dataset_path)
    context = gx.get_context()

    pandas_default = getattr(getattr(context, "sources", None), "pandas_default", None)
    if pandas_default is None:
        pandas_default = context.data_sources.pandas_default

    validator = pandas_default.read_dataframe(frame, asset_name="telco_churn")
    validator.expect_table_row_count_to_be_between(min_value=5_000, max_value=8_000)
    validator.expect_column_values_to_not_be_null("customerID")
    validator.expect_column_values_to_be_in_set("SeniorCitizen", [0, 1])
    validator.expect_column_values_to_be_between("tenure", min_value=0, max_value=72)
    validator.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)
    validator.expect_column_values_to_be_between("TotalCharges", min_value=0, max_value=10_000)
    validator.expect_column_values_to_be_in_set("Churn", ["Yes", "No"])
    result = validator.validate()

    success = getattr(result, "success", None)
    if success is None:
        success = bool(result.get("success", False))
    print(f"Great Expectations validation success: {success}")
    return bool(success)


def main() -> None:
    try:
        success = _great_expectations_validate()
    except ImportError:
        print("great_expectations not installed; using fallback validation.")
        success = _fallback_validate()

    if not success:
        raise SystemExit(1)

    print("Dataset validation complete.")


if __name__ == "__main__":
    main()
