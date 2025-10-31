import pandas as pd

input_file = "1_digital_library_items.csv"

output_file = "1_sampled.csv"

sample_size = 500

df = pd.read_csv(input_file)

sampled_df = df.sample(n=sample_size, random_state=42)
sampled_df.to_csv(output_file, index=False)
