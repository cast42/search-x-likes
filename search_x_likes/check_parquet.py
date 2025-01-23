# Python script to debug column types in parquet file

import pyarrow.parquet as pq
from datasets import load_dataset

# Load the PyArrow table from a Parquet file
table = pq.read_table("/Users/ln/Downloads/train-00000-of-00001.parquet")
# Print the entire schema (all columns)
print(table.schema)

# Inspect just the embeddings column type
embeddings_field = table.schema.field_by_name("embeddings")
print("Embeddings field type:", embeddings_field.type)

# Convert the PyArrow table to a Pandas DataFrame
df = table.to_pandas()

# Inspect the DataFrame
print(df.head())
print(df.info())

print(df.dtypes)

# Load the IMDb dataset
dataset = load_dataset("cast42/x_likes", split="train")

# Convert to Pandas DataFrame
df = dataset.to_pandas()

# Inspect the DataFrame
print(df.head())
