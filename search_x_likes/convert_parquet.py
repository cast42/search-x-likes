# This script is here just for reference
# On itermediate result did not had the correct column types
# The embeddings collumn was op type string.
# I used this script to convert the embeddings column to List[float] type.

import ast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_FILE: str = "./data/liked_posts_embedded.parquet"

df = pd.read_parquet(PARQUET_FILE)
df["embeddings"] = df["embeddings"].map(lambda x: np.array(ast.literal_eval(x)))

#  Schema where `embeddings` is a list of float32
schema = pa.schema([
    pa.field("tweet_id", pa.string()),
    pa.field("full_text", pa.string()),
    pa.field("expanded_url", pa.string()),
    pa.field("embeddings", pa.list_(pa.float32())),
])

# Convert the entire DataFrame to a PyArrow Table
table = pa.Table.from_pandas(df, schema=schema)
pq.write_table(table, "data/embeddings.parquet")

# Read the Parquet file into a PyArrow table
table = pq.read_table("data/embeddings.parquet")

# Print the entire schema (all columns)
print(table.schema)

# Inspect just the embeddings column type
embeddings_field = table.schema.field_by_name("embeddings")
print("Embeddings field type:", embeddings_field.type)
