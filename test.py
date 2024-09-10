from conditional_parser import ConditionalArgumentParser
from ptrseq.utils import argbool


def main():
    parser = ConditionalArgumentParser(description="Task-based argument parser example.")
    parser.add_argument("--bool", default=False, type=argbool)
    args = parser.parse_args()

    print(vars(args))


if __name__ == "__main__":
    main()
