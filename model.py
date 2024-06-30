#pip install transformers
#which python3
#/usr/local/bin/python3 -m venv myenv
#source myenv/bin/activate
#pip install transformers
#pip install torch

# Load model directly
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)


def our_pipeline(payload):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer) 

    res = classifier(payload)

    return res

payload = """
WestBridge Capital sells 1.7% stake in AU Small Finance Bank for Rs 845 crore
Private equity firm WestBridge Capital divested on Wednesday a 1.7 per cent stake in the AU Small Finance Bank for Rs 845 crore through an open market transaction.

WestBridge Capital, through its affiliate Westbridge AIF I, sold shares of the Jaipur-based AU Small Finance Bank through a bulk deal on the National Stock Exchange (NSE).

As per data available, Westbridge AIF I offloaded 1.30 crore shares, amounting to a 1.75 per cent stake in AU Small Finance Bank.

The shares were disposed of at an average price of Rs 650.08 apiece, taking the transaction value to Rs 845.10 crore.

After the stake sale, the shareholding of WestBridge Capital in the AU Small Finance Bank has declined to 2.07 per cent from 3.82 per cent.

Meanwhile, Goldman Sachs Investments Mauritius I picked up 43.34 lakh shares of the AU Small Finance Bank at an average price of Rs 650 per piece. This took the deal value to Rs 281.71 crore.

Details of other buyers of the AU Small Finance Bank shares could not be ascertained.



"""

print(our_pipeline(payload))
