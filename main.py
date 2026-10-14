import argparse

from ExperimentRunner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Please pass path to config file"
    )

    args = parser.parse_args()
    config_path = args.config
    runner = ExperimentRunner(config_path)


if __name__ == "__main__":
    main()
