from argparse import ArgumentParser
from ptrseq.experiments.arglib import add_scheduling_parameters
from ptrseq.utils import scheduler_from_parser

if __name__ == "__main__":
    parser = ArgumentParser(description="ArgumentParser for loading experiment constructor", add_help=False)
    parser
    parser = add_scheduling_parameters(parser, "lr")
    args = parser.parse_args()
    scheduler = scheduler_from_parser(args, "lr", initial_value=1.0)
    print(args)
    print(vars(scheduler))

    print("Testing the scheduler:")
    scheduler.set_epoch(-5)
    for epoch in range(-5, 15):
        print(epoch, scheduler.get_epoch(), scheduler.get_value())
        scheduler.step()
