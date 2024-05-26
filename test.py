from ptrseq.utils import ConditionalArgumentParser
from ptrseq.utils import argbool


def main():
    parser = ConditionalArgumentParser(description="Task-based argument parser example.")
    parser.add_argument("task", type=str, help="The task to run.")
    parser.add_argument("--use_curriculum", type=argbool, default=False)
    parser.add_conditional_argument(
        "task",
        "dominoe",
        "--learning_mode",
        type=str,
        default="supervised",
        help="The learning mode to use.",
    )
    parser.add_conditional_argument(
        "task",
        "dominoe",
        "--randomize_direction",
        type=str,
        default="yes",
        help="whether to use random dominoe directions.",
    )

    parser.add_conditional_argument(
        "use_curriculum",
        True,
        "--curriculum",
        default="standard",
        type=str,
        help="Which curriculum to use",
    )

    parser.add_conditional_argument(
        "curriculum",
        "nonstandard",
        "--curriculum_path",
        default="",
        type=str,
        help="The path to the curriculum file.",
    )

    # Parse the arguments
    # args = ["dominoe", "--use_curriculum", "True", "--curriculum", "nonstandard", "--help"]
    args = parser.parse_args(args=None)

    print(vars(args))


if __name__ == "__main__":
    main()
