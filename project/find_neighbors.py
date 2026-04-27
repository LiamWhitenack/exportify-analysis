from argparse import ArgumentParser
from collections.abc import Iterable
from difflib import get_close_matches

from numpy import dot, ndarray
from numpy.linalg import norm
from pandas import DataFrame, Series, concat
from pandas.api.types import is_datetime64tz_dtype
from sklearn.preprocessing import RobustScaler

from project.read import read_playlists


def keys_go_together(
    key_a: int,
    mode_a: int,
    key_b: int,
    mode_b: int,
    allow_relative: bool = True,
    allow_fifths: bool = True,
) -> bool:
    """
    key: 0–11 (C=0, C#/Db=1, ..., B=11)
    mode: 1=major, 0=minor

    Returns True if the keys are harmonically compatible.
    """

    key_a %= 12
    key_b %= 12

    # 1. Exact match
    if key_a == key_b and mode_a == mode_b:
        return True

    # 2. Relative major/minor (same notes)
    if allow_relative and mode_a != mode_b:
        if mode_a == 1 and key_b == (key_a - 3) % 12:
            return True
        if mode_a == 0 and key_b == (key_a + 3) % 12:
            return True

    # 3. Circle of fifths neighbors (same mode only)
    if allow_fifths and mode_a == mode_b:
        if key_b in {(key_a + 7) % 12, (key_a - 7) % 12}:
            return True

    return False


def time_signatures_match(ts1: int, ts2: int) -> bool:
    if 0 in (ts1, ts2):
        return False
    return (ts1 % ts2 == 0) or (ts2 % ts1 == 0)


def essentials_match(s1: Series, s2: Series) -> bool:
    if (s1["Track Name"], s1["Album Name"]) == (s2["Track Name"], s2["Album Name"]):
        return False
    if time_signatures_match(s1["Time Signature"], s2["Time Signature"]):
        return False
    if not keys_go_together(
        s1["Key"],
        s1["Mode"],
        s2["Key"],
        s2["Mode"],
        allow_relative=True,
        allow_fifths=True,
    ):
        return False

    return tempos_match(s1["Tempo"], s2["Tempo"])


def tempos_match(t1: float, t2: float, tolerance: int = 0.5) -> bool:
    tolerate_digits = 1 / tolerance
    tempo1 = round(t1 * tolerate_digits) / tolerate_digits
    tempo_options = {
        # round(t2 * 1) / tolerate_digits,
        round(t2 * 2) / tolerate_digits,
        # round(t2 * 4) / tolerate_digits,
    }

    return tempo1 in tempo_options


BASE_FEATURES = [
    "Popularity",
    "Danceability",
    "Energy",
    "Loudness",
    "Speechiness",
    "Acousticness",
    "Instrumentalness",
    "Liveness",
    "Valence",
]

INTERACTION_FEATURES = [
    "Acoustic_Vocal",
    "Electronic_Vocal",
    "Acoustic_Instrumental",
    "Electronic_Instrumental",
]

MATCH_FEATURES = BASE_FEATURES + INTERACTION_FEATURES


MATCH_WEIGHTS = {
    "Popularity": 0.5,
    "Danceability": 1.5,
    "Energy": 1.5,
    "Loudness": 0.8,
    "Speechiness": 0.8,
    "Acousticness": 1.2,
    "Instrumentalness": 1.0,
    "Liveness": 0.6,
    "Valence": 1.4,
    "Acoustic_Vocal": 1.2,
    "Electronic_Vocal": 1.0,
    "Acoustic_Instrumental": 1.0,
    "Electronic_Instrumental": 1.0,
}


def add_interactions(df: DataFrame) -> DataFrame:
    df = df.copy()

    df["Acoustic_Vocal"] = df["Acousticness"] * df["Speechiness"]
    df["Electronic_Vocal"] = (1 - df["Acousticness"]) * df["Speechiness"]
    df["Acoustic_Instrumental"] = df["Acousticness"] * df["Instrumentalness"]
    df["Electronic_Instrumental"] = (1 - df["Acousticness"]) * df["Instrumentalness"]

    return df


def vector(s: Series) -> ndarray:
    return s[MATCH_FEATURES].astype(float).to_numpy()


def find_compatible_songs(
    candidates: DataFrame,
    target: Series,
) -> DataFrame:
    # filter by your existing logic
    candidates = candidates.copy()[
        candidates.apply(lambda candidate: essentials_match(target, candidate), axis=1)
    ]

    if candidates.empty:
        return candidates

    # add interaction features
    candidates = add_interactions(candidates)
    target_df = add_interactions(target.to_frame().T)

    # prepare weights
    feature_weights = (
        Series(MATCH_WEIGHTS).reindex(MATCH_FEATURES).astype(float).to_numpy()
    )

    # scale features robustly
    scaler = RobustScaler()

    candidate_values = candidates[MATCH_FEATURES].astype(float)
    target_values = target_df[MATCH_FEATURES].astype(float)

    scaled_candidates = scaler.fit_transform(candidate_values)
    scaled_target = scaler.transform(target_values)[0]

    # weighted distance
    weighted_diff = (scaled_candidates - scaled_target) * feature_weights
    scores = norm(weighted_diff, axis=1)

    candidates["Score"] = scores

    return candidates.sort_values("Score").drop_duplicates(
        ["Track Name", "Artist Name(s)"]
    )


def find_song(
    df: DataFrame,
    song_name: str,
) -> Series:
    df = df.drop_duplicates(["Track Name", "Album Name"])
    matches = [
        (i, row["Track Name"], row["Album Name"], row["Artist Name(s)"])
        for i, row in df.iterrows()
        if song_name.lower().replace(" ", "")
        in row["Track Name"].lower().replace(" ", "")
    ]

    if not matches:
        suggestions = get_close_matches(
            song_name,
            set(df["Track Name"].apply(lambda p: p.lower().replace(" ", ""))),
            n=5,
            cutoff=0.4,
        )

        raise Exception(
            f"No matches found for '{song_name}'. Did you mean: {suggestions}?"
        )

    if len(matches) == 1:
        return df.loc[matches[0][0]]

    print("Matches found:")
    for option_number, (_, track_name, album, artist) in enumerate(matches, start=1):
        print(f"{option_number}: {track_name} ({album, artist})")

    while True:
        choice = input(f"Select 1-{len(matches)} (or 'q' to quit): ").strip()

        if choice.lower() == "q":
            raise Exception("User quit song selection.")

        if choice.isdigit():
            selected_option = int(choice)
            if 1 <= selected_option <= len(matches):
                selected_index = matches[selected_option - 1][0]
                return df.loc[selected_index]

        print("Invalid choice. Try again.")


def remove_timezones(df: DataFrame) -> DataFrame:
    df = df.copy()

    for column in df.columns:
        if is_datetime64tz_dtype(df[column]):
            df[column] = df[column].dt.tz_localize(None)

    return df


if __name__ == "__main__":
    df = read_playlists()
    matches = DataFrame()
    dfs = []
    for i, (_, row) in enumerate(df.iterrows()):
        if (matches := find_compatible_songs(df, row)).empty:
            continue
        matches["Matched To Track"] = row["Track Name"]
        matches["Matched To Artist"] = row["Artist Name(s)"]
        dfs.append(matches)
        if i > len(df) / 2:
            break
        print(f"found match! {i}/{len(df)}")
    remove_timezones(concat(dfs))[
        [
            "Track Name",
            "Artist Name(s)",
            "Matched To Track",
            "Matched To Artist",
            "Score",
        ]
    ].sort_values(["Score"]).to_excel("matches.xlsx")
    while True:
        matches = find_compatible_songs(
            df, find_song(df, input("Search for match (song name):"))
        )
        if matches.empty:
            print("\nNo Matches Found!\n")
        else:
            print("\n", matches, "\n")
    # for _, row in df.iterrows():
    #     compatible = find_compatible_songs(df, row)
    #     if not compatible.empty:
    #         print(row["Track Name"])
    #         print(compatible)
    #         break
    pass
