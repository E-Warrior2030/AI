import pandas as pd
import time

def clean_chunk(df):
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.drop_duplicates()
    return df

def process_large_file(path_in, path_out, chunksize=100_000):
    start = time.time()
    first = True
    total_rows = 0

    for chunk in pd.read_csv(path_in, chunksize=chunksize):
        t0 = time.time()
        clean = clean_chunk(chunk)
        clean.to_csv(path_out, mode='w' if first else 'a', index=False, header=first)
        first = False
        total_rows += len(clean)
        print(f"Chunk rows: {len(clean)}, time: {time.time() - t0:.3f}s")

    print(f"Total rows written: {total_rows}")
    print(f"Total time: {time.time() - start:.3f}s")
