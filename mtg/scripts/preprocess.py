import argparse
from mtg.obj.expansion import VOW
import pickle


def main():
    expansion = VOW(bo1=FLAGS.game_data, draft=FLAGS.draft_data, ml_data=True,)
    with open(FLAGS.expansion_fname, "wb") as f:
        pickle.dump(expansion, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game_data", type=str, default=None, help="path to bo1 game data"
    )
    parser.add_argument(
        "--draft_data", type=str, default=None, help="path to bo1 draft data"
    )
    parser.add_argument(
        "--expansion_fname",
        type=str,
        default="expansion.pkl",
        help="path/to/fname.pkl for where we should store the expansion object",
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()
