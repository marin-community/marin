# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small helpers for sqrt(width) learning-rate fits in speedrun sweeps."""

import math


def r2_rmse(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    rmse = math.sqrt(ss_res / len(y_true))
    return r2, rmse


def fit_lr_vs_sqrt_width(points):
    widths = list(points)
    lrs = list(points.values())
    sqrt_widths = [math.sqrt(w) for w in widths]
    mean_x = sum(sqrt_widths) / len(sqrt_widths)
    mean_y = sum(lrs) / len(lrs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(sqrt_widths, lrs)) / sum(
        (x - mean_x) ** 2 for x in sqrt_widths
    )
    intercept = mean_y - slope * mean_x
    preds = [intercept + slope * x for x in sqrt_widths]
    r2, rmse = r2_rmse(lrs, preds)
    return intercept, slope, r2, rmse


def predict_lr_from_width(intercept, slope, width):
    return intercept + slope * math.sqrt(width)


def width_lr_fit():
    width_points = {
        512: 0.008,
        768: 0.008,
        1024: 0.008,
        2048: 0.006,
    }
    return fit_lr_vs_sqrt_width(width_points)


def main():
    intercept, slope, r2, rmse = width_lr_fit()
    print(
        "width fit: lr = a + b * sqrt(hidden_dim) "
        f"(a={intercept}, b={slope}, r2={r2}, rmse={rmse})"
    )
    for width in (640, 896, 1408):
        lr = predict_lr_from_width(intercept, slope, width)
        print(f"{width}: lr={round(lr, 6)}")


if __name__ == "__main__":
    main()
