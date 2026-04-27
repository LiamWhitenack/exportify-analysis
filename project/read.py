import os

from pandas import DataFrame, concat, read_csv


def read_playlists() -> DataFrame:
    dfs: list[DataFrame] = []
    for file in os.listdir("data"):
        df = read_csv("data/" + file, engine="pyarrow")
        df["Playlist"] = file[:-4]
        dfs.append(df)
    return (
        concat(dfs).drop_duplicates(["Track Name", "Artist Name(s)"]).reset_index()
    )  # .drop_duplicates(["Track Name", "Album Name"])


if __name__ == "__main__":
    print(read_playlists())
    print(read_playlists().columns)
