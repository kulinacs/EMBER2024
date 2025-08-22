import argparse
import pathlib

import lightgbm as lgb

import thrember


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_directory",
                        type=pathlib.Path,
                        help="path to save the pre-trained models",
                        default=pathlib.Path(
                            __file__).resolve().parent / "models",
                        )
    parser.add_argument("--model",
                        type=str,
                        help="model to use for evaluation",
                        default="PE",
                        )
    parser.add_argument("--threshold",
                        type=int,
                        help="a threshold to return an error exit code if passed",
                        default=101,
                        )
    parser.add_argument("sample",
                        type=pathlib.Path,
                        help="the sample to evaluate",
                        )
    args = parser.parse_args()

    if not args.model_directory.exists():
        args.model_directory.mkdir(parents=True)

    thrember.download_models(args.model_directory)

    model_file = args.model_directory / f"EMBER2024_{args.model}.model"
    model = lgb.Booster(model_file=model_file)

    with args.sample.open("rb") as f:
        file_data = f.read()

    prediction = thrember.predict_sample(model, file_data)
    print(prediction * 100)

    if prediction > args.threshold:
        exit(1)


if __name__ == "__main__":
    main()
