import argparse
import os

import numpy as np
import itertools

try:
    from marin.utilities.wandb_utils import WANDB_ENTITY, WANDB_PROJECT
except Exception:
    WANDB_ENTITY = os.getenv("WANDB_ENTITY", "stanford-mercury")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "marin")

from experiments.regmix.plot_results import (
    MIXTURE_TAG_KEYS,
    collect_regmix_summary,
    METRICS,
)


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM regressor on regmix mixtures to predict a metric.")
    parser.add_argument("--entity", default=WANDB_ENTITY, help="W&B entity")
    parser.add_argument("--project", default=WANDB_PROJECT, help="W&B project")
    parser.add_argument("--api_key", default=os.getenv("WANDB_API_KEY"), help="W&B API key")
    parser.add_argument(
        "--metric_slug",
        default="bpb",
        choices=list(METRICS.keys()),
        help="Which metric to predict: one of {" + ", ".join(METRICS.keys()) + "}",
    )
    parser.add_argument(
        "--require_tags",
        action="append",
        default=[
            "regmix",
            "llama-130m",
            "lr=0.006",
        ],
        help="Required tags to include runs (repeatable)",
    )
    parser.add_argument(
        "--also_accept_tag",
        action="append",
        default=["130m"],
        help="Fallback tags to accept if some required tags are missing (repeatable)",
    )
    parser.add_argument("--name_contains", default="llama-130m-regmix-v3")
    parser.add_argument("--test_size", type=float, default=0.10)
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="Run a simple grid search over key LightGBM hyperparameters using the validation split",
    )
    args = parser.parse_args()

    rows = collect_regmix_summary(
        entity=args.entity,
        project=args.project,
        api_key=args.api_key,
        require_tags=list(args.require_tags or []),
        also_accept_tag=list(args.also_accept_tag or []),
        name_contains=args.name_contains,
    )

    if not rows:
        raise SystemExit("No runs found for regression.")

    # Build dataset
    X_list: list[list[float]] = []
    y_list: list[float] = []

    for r in rows:
        mixture = r["mixture"]
        metrics_map = r["metrics"]
        y_val = metrics_map.get(args.metric_slug)
        if y_val is None:
            continue
        X_list.append([float(mixture.get(k, 0.0)) for k in MIXTURE_TAG_KEYS])
        y_list.append(float(y_val))

    if not X_list:
        raise SystemExit("No rows with the target metric available.")

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)

    # Train/valid split
    from sklearn.model_selection import train_test_split

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=42)

    # Train LightGBM regressor (optionally with a grid search over key hyperparameters)
    import lightgbm as lgb

    model = None
    best_params = None
    best_valid_rmse = float("inf")
    best_valid_r2 = float("-inf")

    if args.grid_search:
        # Keep the grid compact to avoid explosion, focusing on impactful knobs
        param_grid = {
            "learning_rate": [0.02, 0.05, 0.1],
            "num_leaves": [16, 31],
            "max_depth": [4, 6],
            "min_child_samples": [5, 20],
            "reg_alpha": [0.0, 1.0],
            "reg_lambda": [0.0, 1.0],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "n_estimators": [400, 700],
        }

        keys = list(param_grid.keys())
        combos = list(itertools.product(*[param_grid[k] for k in keys]))

        from sklearn.metrics import mean_squared_error, r2_score

        print(f"Grid search: evaluating {len(combos)} combinations...")
        for values in combos:
            params = dict(zip(keys, values, strict=False))
            candidate = lgb.LGBMRegressor(
                random_state=42,
                n_jobs=-1,
                **params,
            )
            candidate.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="l2",
            )

            best_iter = getattr(candidate, "best_iteration_", None)
            y_pred_valid = candidate.predict(X_valid, num_iteration=best_iter)
            rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred_valid)))
            r2 = float(r2_score(y_valid, y_pred_valid))

            if rmse < best_valid_rmse:
                best_valid_rmse = rmse
                best_valid_r2 = r2
                best_params = params
                model = candidate

        print("Best grid search results:")
        print(f"  RMSE: {best_valid_rmse:.6f}")
        print(f"  R2:   {best_valid_r2:.6f}")
        if best_params is not None:
            print("  Params:")
            for k in sorted(best_params.keys()):
                print(f"    {k} = {best_params[k]}")
    else:
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=16,
            min_child_samples=5,  # at least 5 samples per leaf
            reg_alpha=1.0,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="l2",
            # callbacks=[lgb.early_stopping(20)],
        )

    # Report metrics
    from sklearn.metrics import mean_squared_error, r2_score

    train_pred = model.predict(X_train, num_iteration=getattr(model, "best_iteration_", None))
    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))
    train_r2 = float(r2_score(y_train, train_pred))
    print(f"Train RMSE: {train_rmse:.6f}")
    print(f"Train R2:   {train_r2:.6f}")

    best_iter = getattr(model, "best_iteration_", None)
    y_pred = model.predict(X_valid, num_iteration=best_iter)
    rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred)))
    r2 = float(r2_score(y_valid, y_pred))

    print(f"Metric: {args.metric_slug}")
    if best_iter is not None:
        print(f"Best iteration: {best_iter}")
    print(f"Valid RMSE: {rmse:.6f}")
    print(f"Valid R2:   {r2:.6f}")

    # Feature importances
    importances = model.feature_importances_
    for key, imp in zip(MIXTURE_TAG_KEYS, importances, strict=False):
        print(f"{key}: {imp}")


if __name__ == "__main__":
    main()
