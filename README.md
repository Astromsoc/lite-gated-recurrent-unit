# Lite GRU for Inference

Personal implementation of light-weight gated recurrent unit inference for simply encoded, simply decoded seq2seq or regression/classification style of tasks.

## Environmental Setup

```bash
source setup.sh
```

## Model Training

```bash
python3 -m src.train \
        # filepath to configurations
        -f "cfgs/sample-gppred-train.yaml"
```

### Best Architecture Hitherto

```json
{
  "enc_emb_dim": 128,
  "dec_emb_dim": 128,
  "encgru": {
    "hidden_size": 1024,
    "batch_first": true
  },
  "enc_take_last": false,
  "decgru": {
    "hidden_size": 1024
  },
  "cls_lin_dim": 1024,
  "max_steps": 40
}
```
...which has achieved a Levenshtein distance of `0.00` on training set and `0.4277` on validation set.


## Model Inference

### Local Test with Simple Termial Inputs

```bash
python3 -m src.infer \
        # filepath to checkpoint
        -c "<filepath to model checkpoint>"
```

### API Calls

Launch the deployed model checkpoint by running:

```bash
uvicorn api.infer_api:app --reload
```

...and then send requests (modify samples [here](api/test_client.py)) by running:

```bash
python3 -m api.test_client
```
