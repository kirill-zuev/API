import argparse


def make_parser():
    parser = argparse.ArgumentParser("parameters")
    # parser.add_argument("--filter", default="4", help="filter number")
    parser.add_argument("--lang", default="ru", help="language")
    return parser