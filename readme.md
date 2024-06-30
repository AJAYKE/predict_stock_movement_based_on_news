## Predict stock movement with news article

We are using "ProsusAI/finbert" model.

first we scrape the data from the moneycontrol website

setup a private environment on ur local

and then run the notebook

```bash
which python3
```

```bash
/usr/local/bin/python3 -m venv myenv
```

```bash
source myenv/bin/activate
```

we are using pandas, beautifulsoup4, transformers and pytorch(required for finbert) libraries.

initially we will scrape the data, then we download the model from huggingface using transformers library, then we run the model for which pytorch is needed
