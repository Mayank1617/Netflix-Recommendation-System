"""Data Processing Pipeline.

Parses the raw Netflix Prize `combined_data_*.txt` files, builds a dense
size-capped subset, and produces a temporal train/test split.

The functions here mirror Stages 1-3 of `netflix_recsys.ipynb` so the repo can
be run either as a notebook (Colab) or as importable modules.
"""
from __future__ import annotations
import glob
import gc
import numpy as np
import pandas as pd


def parse_combined(path: str) -> pd.DataFrame:
    """Parse one `combined_data_*.txt` file into a (user, movie, rating, date) frame.

    A line like ``123:`` marks movie 123; the following ``CustomerID,Rating,Date``
    rows belong to it until the next marker. We forward-fill the movie id rather
    than looping line by line.
    """
    df = pd.read_csv(path, header=None, names=["user", "rating", "date"])
    is_movie = df["user"].str.endswith(":")
    movie = df["user"].where(is_movie).str.rstrip(":").ffill()
    keep = ~is_movie
    return pd.DataFrame({
        "user":   df.loc[keep, "user"].astype(np.int32).to_numpy(),
        "movie":  movie[keep].astype(np.int32).to_numpy(),
        "rating": df.loc[keep, "rating"].astype(np.int8).to_numpy(),
        "date":   pd.to_datetime(df.loc[keep, "date"]).to_numpy(),
    })


def parse_to_parquet(data_dir: str = "netflix_data") -> pd.Series:
    """PASS 1 — parse each file to its own parquet (low peak RAM); return movie counts.

    Writing each file to disk and freeing it keeps peak memory at ~one file
    instead of holding all 100M ratings at once (the free-tier OOM fix).
    """
    paths = sorted(glob.glob(f"{data_dir}/combined_data_*.txt"))
    counts = []
    for i, p in enumerate(paths):
        part = parse_combined(p)
        part.to_parquet(f"part_{i}.parquet", index=False)
        counts.append(part["movie"].value_counts())  # each movie lives in one file
        del part
        gc.collect()
    return pd.concat(counts)


def build_subset(movie_counts: pd.Series, n_paths: int,
                 n_top_movies: int = 3000, n_users_sample: int = 40000,
                 min_user_ratings: int = 50, seed: int = 42) -> pd.DataFrame:
    """PASS 2 — keep top-N popular movies, then sample active users (reproducible)."""
    top_movies = set(movie_counts.sort_values(ascending=False).head(n_top_movies).index)
    frames = []
    for i in range(n_paths):
        part = pd.read_parquet(f"part_{i}.parquet")
        frames.append(part[part["movie"].isin(top_movies)])
        del part
        gc.collect()
    ratings = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    uc = ratings["user"].value_counts()
    active = uc[uc >= min_user_ratings].index.to_numpy()
    rng = np.random.default_rng(seed)
    sampled = rng.choice(active, size=min(n_users_sample, len(active)), replace=False)
    return ratings[ratings["user"].isin(sampled)].reset_index(drop=True)


def temporal_split(ratings: pd.DataFrame, test_fraction: float = 0.2):
    """Hold out each user's most recent `test_fraction` of ratings (no leakage)."""
    ratings = ratings.sort_values(["user", "date"]).reset_index(drop=True)
    grp = ratings.groupby("user")
    ratings["rank"] = grp.cumcount()
    ratings["size"] = grp["rating"].transform("size")
    cutoff = np.ceil(ratings["size"] * (1 - test_fraction)).astype(int)
    test = ratings[ratings["rank"] >= cutoff].drop(columns=["rank", "size"]).reset_index(drop=True)
    train = ratings[ratings["rank"] < cutoff].drop(columns=["rank", "size"]).reset_index(drop=True)
    return train, test


def build_index(train: pd.DataFrame):
    """Contiguous index maps over the train universe of users/items."""
    u_ids = np.sort(train["user"].unique())
    m_ids = np.sort(train["movie"].unique())
    u2i = {int(u): i for i, u in enumerate(u_ids)}
    m2i = {int(m): i for i, m in enumerate(m_ids)}
    return u_ids, m_ids, u2i, m2i


def map_df(df: pd.DataFrame, u2i: dict, m2i: dict):
    """Map a ratings frame to (user_idx, item_idx, rating) numpy arrays."""
    df = df[df["user"].isin(u2i) & df["movie"].isin(m2i)]
    return (df["user"].map(u2i).to_numpy(),
            df["movie"].map(m2i).to_numpy(),
            df["rating"].astype(np.float32).to_numpy())
