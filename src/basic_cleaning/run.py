#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Download input artifact from W&B
    logger.info("Downloading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Read into DataFrame
    logger.info("Reading input data")
    df = pd.read_csv(artifact_local_path)

    # Drop outliers based on price
    logger.info("Filtering rows based on price limits")
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    # Drop rows with missing values in certain columns (optional, adjust as needed)
    logger.info("Dropping rows with missing values in 'name' and 'last_review' columns")
    df = df.dropna(subset=["name", "last_review"])

    # Save cleaned data to a local CSV file
    cleaned_file = "clean_sample.csv"
    logger.info(f"Saving cleaned data to {cleaned_file}")
    df.to_csv(cleaned_file, index=False)

    # Log cleaned data as a new artifact
    logger.info("Logging cleaned data as new W&B artifact")
    artifact = wandb.Artifact(
    args.output_artifact,
    type=args.output_type,
    description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

    logger.info("Basic cleaning step completed successfully.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Fully-qualified name for the input artifact (e.g., 'sample.csv:latest')",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name for the cleaned data artifact to be created",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type for the output artifact (e.g., 'clean_data')",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description for the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum listing price to include",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum listing price to include",
        required=True
    )


    args = parser.parse_args()

    go(args)
