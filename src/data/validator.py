"""Data validation module using Great Expectations.

Validates cleaned data against quality expectations.
"""

import great_expectations as gx
from great_expectations.core import ExpectationSuite

from src.config import get_config
from src.data.cleaner import clean_data
from src.data.loader import load_raw_data
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_data() -> bool:
    """Run Great Expectations validation on cleaned dataset.

    Returns:
        True if all critical expectations pass.
    """
    config = get_config()
    raw_df = load_raw_data(config)
    df = clean_data(raw_df, config)

    logger.info("validation_started", rows=len(df))

    context = gx.get_context()

    data_source = context.data_sources.add_pandas("pandas_source")
    data_asset = data_source.add_dataframe_asset(name="imdb_reviews")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "full_batch"
    )
    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": df}
    )

    suite = ExpectationSuite(name="imdb_data_quality")

    # Target column values must be 0 or 1
    suite.add_expectation(
        gx.expectations.ExpectColumnDistinctValuesToBeInSet(
            column=config.data.target_column,
            value_set=[0, 1],
        )
    )

    # Review column must have zero nulls
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column=config.data.text_column,
        )
    )

    # ds_part values must be train or test
    suite.add_expectation(
        gx.expectations.ExpectColumnDistinctValuesToBeInSet(
            column=config.data.split_column,
            value_set=["train", "test"],
        )
    )

    # Class balance: positive class between 45% and 55%
    suite.add_expectation(
        gx.expectations.ExpectColumnMeanToBeBetween(
            column=config.data.target_column,
            min_value=0.45,
            max_value=0.55,
        )
    )

    # Review length within bounds
    suite.add_expectation(
        gx.expectations.ExpectColumnValueLengthsToBeBetween(
            column=config.data.text_column,
            min_value=config.data.min_review_length,
        )
    )

    all_passed = True
    for expectation in suite.expectations:
        result = batch.validate(expectation)
        exp_type = type(expectation).__name__
        passed = result.success
        if not passed:
            all_passed = False
            logger.warning("expectation_failed", expectation=exp_type)
        else:
            logger.info("expectation_passed", expectation=exp_type)

    if all_passed:
        logger.info("all_expectations_passed")
    else:
        logger.warning("some_expectations_failed")

    return all_passed


if __name__ == "__main__":
    validate_data()
