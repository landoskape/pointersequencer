import argparse
from ptrseq.experiments.arglib import add_dataset_parameters


def main():
    parser = argparse.ArgumentParser(description="Task-based argument parser example.")
    parser = add_dataset_parameters(parser)

    # Parse the arguments
    args = parser.parse_args()

    print(vars(args))


if __name__ == "__main__":
    main()
