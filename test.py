from conditional_parser import ConditionalArgumentParser
from ptrseq.utils import argbool, argintrange


def main():
    parser = ConditionalArgumentParser(description="Task-based argument parser example.")
    parser.add_argument("--none-or-2", default=None, nargs=2, type=argintrange)
    args = parser.parse_args()

    print(vars(args))


if __name__ == "__main__":
    main()
