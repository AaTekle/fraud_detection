# src/eda.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Creating the reports folder if it does not already exist.
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv("data/creditcard.csv")

# Printing out the dataset size and class balance.
print(df.shape)
print(df["Class"].value_counts())
# Printing out the the proportion of each Class value.
print(df["Class"].value_counts(normalize=True))

# Creating and save a bar chart showing the number of transactions in each class.
plt.figure()
df["Class"].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig(REPORT_DIR / "class_distribution.png", bbox_inches="tight")
plt.close()

# Creating and save a histogram showing the distribution of transaction amounts.
plt.figure()
df["Amount"].plot(kind="hist", bins=100)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.savefig(REPORT_DIR / "amount_distribution.png", bbox_inches="tight")
plt.close()
