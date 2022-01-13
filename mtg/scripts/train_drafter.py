import argparse
from mtg.ml.generator import DraftGenerator, create_train_and_val_gens
from mtg.ml.models import DraftBot
import pickle
from mtg.ml.trainer import Trainer
from mtg.utils.display import draft_log_ai


def main():
    with open(FLAGS.expansion_fname, "rb") as f:
        expansion = pickle.load(f)

    train_gen, val_gen = create_train_and_val_gens(
        expansion.draft,
        expansion.cards.copy(),
        train_p=FLAGS.train_p,
        id_col="draft_id",
        train_batch_size=FLAGS.batch_size,
        generator=DraftGenerator,
        include_val=True,
    )

    model = DraftBot(
        expansion=expansion,
        emb_dim=FLAGS.emb_dim,
        num_encoder_heads=FLAGS.num_encoder_heads,
        num_decoder_heads=FLAGS.num_decoder_heads,
        pointwise_ffn_width=FLAGS.pointwise_ffn_width,
        num_encoder_layers=FLAGS.num_encoder_layers,
        num_decoder_layers=FLAGS.num_decoder_layers,
        emb_dropout=FLAGS.emb_dropout,
        memory_dropout=FLAGS.transformer_dropout,
        name="DraftBot",
    )

    model.compile(
        learning_rate={"warmup_steps": FLAGS.lr_warmup},
        margin=FLAGS.emb_margin,
        emb_lambda=FLAGS.emb_lambda,
        rare_lambda=FLAGS.rare_lambda,
        cmc_lambda=FLAGS.cmc_lambda,
    )

    trainer = Trainer(model, generator=train_gen, val_generator=val_gen,)
    trainer.train(
        FLAGS.epochs,
        print_keys=["prediction_loss", "embedding_loss", "rare_loss", "cmc_loss"],
        verbose=FLAGS.verbose,
    )
    # we run inference once before saving the model in order to serialize it with the right input parameters for inference
    output_df, attention = draft_log_ai(
        "https://www.17lands.com/draft/79dcc54822204a20a88a0e68ec3f8564",
        model,
        expansion,
    )
    model.save(FLAGS.model_name)


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
        "--emb_dim", type=int, default=128, help="card embedding dimension"
    )
    parser.add_argument(
        "--num_encoder_heads",
        type=int,
        default=8,
        help="number of heads in the encoder blocks of transformer",
    )
    parser.add_argument(
        "--num_decoder_heads",
        type=int,
        default=8,
        help="number of heads in the decoder blocks of transformer",
    )
    parser.add_argument(
        "--pointwise_ffn_width",
        type=int,
        default=512,
        help="each transformer block has a pointwise_ffn with this width as latent space",
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=2,
        help="number of transformer blocks for the encoder",
    )
    parser.add_argument(
        "--num_decoder_layers",
        type=int,
        default=2,
        help="number of transformer blocks for the decoder",
    )
    parser.add_argument(
        "--emb_dropout",
        type=float,
        default=0.0,
        help="dropout rate to apply to embeddings before passed to encoder",
    )
    parser.add_argument(
        "--transformer_dropout",
        type=float,
        default=0.1,
        help="dropout rate inside each transformer block",
    )
    parser.add_argument(
        "--lr_warmup",
        type=float,
        default=2000,
        help="number of warmup steps in the classic transformer learning rate scheduler",
    )
    parser.add_argument(
        "--emb_margin",
        type=float,
        default=1.0,
        help="margin for triplet loss penalty on the embeddings",
    )
    parser.add_argument(
        "--emb_lambda",
        type=float,
        default=0.5,
        help="regularization coefficient for triplet loss on embeddings",
    )
    parser.add_argument(
        "--rare_lambda",
        type=float,
        default=10.0,
        help="regularization coefficient for penalizing the model for taking rares when human doesn't",
    )
    parser.add_argument(
        "--cmc_lambda",
        type=float,
        default=0.1,
        help="regularization coefficient for penalizing the model for taking expensive cards when human doesn't",
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
        help="path/to/draft_model where the model will be stored",
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
