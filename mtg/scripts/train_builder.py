import argparse
from mtg.ml.generator import DeckGenerator, create_train_and_val_gens
from mtg.ml.models import DeckBuilder
import pickle
from mtg.ml.trainer import Trainer
import numpy as np
from mtg.utils.display import build_decks
from mtg.ml.utils import load_model


def main():
    with open(FLAGS.expansion_fname, "rb") as f:
        expansion = pickle.load(f)

    decks = expansion.get_bo1_decks()
    train_gen, val_gen = create_train_and_val_gens(
        decks,
        expansion.cards,
        train_p=FLAGS.train_p,
        id_col="draft_id",
        train_batch_size=FLAGS.batch_size,
        generator=DeckGenerator,
        include_val=True,
        mask_decks=True,
    )

    if FLAGS.draft_model is not None:
        _, attrs = load_model(FLAGS.draft_model)
        embeddings = attrs["embeddings"]
    else:
        embeddings = FLAGS.emb_dim

    model = DeckBuilder(
        expansion.cards["idx"].max() - 4,
        dropout=FLAGS.dropout,
        embeddings=embeddings,
        name="DeckBuilder",
    )

    model.compile(
        learning_rate={"warmup_steps": FLAGS.lr_warmup},
        cmc_lambda=FLAGS.cmc_lambda,
        card_data=expansion.card_data_for_ML.iloc[:-1, :],
    )
    trainer = Trainer(model, generator=train_gen, val_generator=val_gen,)
    trainer.train(
        FLAGS.epochs, verbose=FLAGS.verbose,
    )
    # we run inference once before saving the model in order to serialize it with the right input parameters for inference
    # and we do it with train_gen because val_gen can be None, and this isn't used for validation but serialization
    x, y, z = train_gen[0]
    pid = 0
    pool = np.expand_dims(x[0][pid, 0, :], 0)
    basics, spells, n_basics = build_decks(model, pool, cards=expansion.cards)

    model.save(expansion.cards, FLAGS.model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expansion_fname",
        type=str,
        default="expansion.pkl",
        help="path/to/fname.pkl for where we should load the expansion object",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="training batch size"
    )
    parser.add_argument(
        "--train_p", type=float, default=1.0, help="number in [0,1] for train-val split"
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=None,
        help="card embedding dimension. If None, embeddings aren't used. If we pass a Draft Model in --draft_model, we use the embeddings from that model instead.",
    )
    parser.add_argument(
        "--draft_model",
        type=str,
        default=None,
        help="path/to/model so we can use embeddings learned from an existing pretrained draft model",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="dropout rate to apply to the dense layers in the encoders",
    )
    parser.add_argument(
        "--lr_warmup",
        type=float,
        default=2000,
        help="number of warmup steps in the classic transformer learning rate scheduler",
    )
    parser.add_argument(
        "--cmc_lambda",
        type=float,
        default=0.1,
        help="regularization coefficient for helping the model build comparable curves to humans",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train the model"
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="If True, tqdm will display during training",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="draft_model",
        help="path/to/deck_model where the model will be stored",
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
