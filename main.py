import argparse
from PIPELINE.pipeline import Pipeline
from TOOLS.config_parser import ConfigParser
from PREPROC.preproc import Preproc


def main():
    # region BLOCK FOR PARSING
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Please pass path to config file"
    )

    args = parser.parse_args()

    PATH = args.config
    # endregion

    # RUN PIPELINE
    config = ConfigParser.parse(PATH)
    Pipeline.run(config)

    p = Preproc
    p.face_detector


if __name__ == "__main__":
    main()
