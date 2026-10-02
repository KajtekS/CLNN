import argparse
from PIPELINE.pipeline import Pipeline
from TOOLS.config_parser import ConfigParser

def main():
    # region BLOCK FOR PARSING
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Please pass path to config file"
    )

    args = parser.parse_args()

    PATH = args.config
    # endregion
    
    # region RUN PIPELINE
    config = ConfigParser.parse(PATH)
    Pipeline.run(config)

if __name__ == "__main__":
    main()