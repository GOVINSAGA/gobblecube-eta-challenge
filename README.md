# Gobblecube ETA Challenge

This is my submission for Challenge 1: predicting NYC Yellow Taxi trip duration
from pickup zone, dropoff zone, request timestamp, and passenger count.

## Result

The starter XGBoost baseline measured `353.7s` MAE on the full Dev split on my
machine. The route-prior model in this repo measured:

| Model artifact | Full Dev MAE | Notes |
| --- | ---: | --- |
| Train-only validation model | `253.998s` | Honest comparison: trained on `train.parquet`, scored on all of `dev.parquet`. |
| Final saved `model.pkl` | `237.311s` | Refit on all distributed 2023 labels (`train+dev`) after model selection. This is not an honest Dev estimate. |

The honest improvement is `99.7s` MAE, a `28.2%` reduction from the starter
baseline. I use the train-only number as the fair validation signal.

I expect the honest number to be the useful local signal. The final artifact is
refit on all non-eval data because the held-out set is a 2024 winter-holiday
slice and late-December 2023 labels are directly relevant history.

Docker verification:

```bash
docker build -t gobblecube-eta .
docker run --rm -v "$(pwd)/data:/work" gobblecube-eta /work/dev.parquet /work/preds.csv
```

The built image was `686MB` locally, well below the `2.5GB` limit.

## Approach

The baseline GBT treats zone IDs as ordered integers, which leaves a lot of
signal unused. The strongest first move was to model each route directly:

- `pickup_zone + dropoff_zone` median duration is already much better than the baseline.
- `route + hour_of_week` median captures commuting and nightlife patterns.
- Recent December route/time medians are blended back to the full-year table so sparse routes do not overfit.
- A small ISO-week residual correction helps with winter-holiday seasonality.

The final prediction is a weighted blend of three robust lookup components:

```text
0.50 * recent route/hour-of-week median
0.20 * recent route/hour median
0.30 * full-year route/hour-of-week median plus week residual
```

Each granular table is shrunk toward a broader fallback using its historical
count. This keeps common Midtown/JFK routes very specific while rare routes fall
back to stable route or zone priors.

## What I Tried

The quick experiments were intentionally simple and measurable:

| Experiment | Full Dev MAE |
| --- | ---: |
| Starter XGBoost baseline | `353.7s` |
| Global median | `545.8s` |
| Route mean | `301.3s` |
| Route median | `296.7s` |
| Route + hour-of-week median, shrunk | `257.0s` |
| December route/time ensemble | `254.0s` |

What worked:

- Route medians were a stronger prior than treating zone IDs as numeric features.
- Medians beat means because trip durations have long delay tails.
- Hour-of-week buckets captured commute/nightlife timing without a heavy model.
- Count-based shrinkage kept sparse route/time cells from overfitting.
- December-only recency improved the late-December Dev split when blended back to full-year priors.

What did not help enough:

- The starter raw-zone GBT underfit route identity and lost to simple route medians.
- Global and time-only priors were too coarse for NYC route variation.
- Passenger-count route buckets barely moved MAE and mostly added sparsity.
- Very granular recent tables helped common routes but became noisy without shrinkage.

I did not keep XGBoost in the final runtime because the lookup model was both
more accurate and much smaller. The shipped `model.pkl` is a plain dictionary of
numpy arrays, so inference is just timestamp parsing plus three sparse table
lookups.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python data/download_data.py
python train_model.py
python -m pytest tests
python grade.py
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

`python train_model.py` first prints the honest train-only Dev MAE, then refits
and saves `model.pkl` on `train+dev`. Use `python train_model.py --honest-only`
if you want the saved artifact to exactly match the validation setup.

## Files

- `predict.py`: required `predict(request: dict) -> float` interface.
- `train_model.py`: reproducible trainer for the lookup artifact.
- `model.pkl`: final trained lookup tables.
- `grade.py`: starter scoring harness.
- `Dockerfile`: grader-style container entrypoint.
- `Agents.md` and `Claude.md`: AI-tooling notes requested by the prompt.

## Next Experiments

If I had another iteration, I would add public zone centroid features for better
rare-route fallback, then train a small residual model on rolling time splits
using only out-of-fold route priors. I would also test public weather signals,
but I avoided them here to keep the runtime deterministic and the Docker image
small.
