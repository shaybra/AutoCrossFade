import os

import librosa
import numpy as np


def analyze_song(file_path):
    y, sr = librosa.load(file_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    key = librosa.key.estimate_tuning(y=y, sr=sr)
    energy = np.mean(librosa.feature.rms(y=y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)
    return {
        'file_path': file_path,
        'tempo': tempo,
        'key': key,
        'energy': energy,
        'mfcc': mfcc,
        'beat_frames': beat_frames
    }


def evaluate_transition(song1, song2):
    tempo_diff = abs(song1['tempo'] - song2['tempo'])
    key_diff = abs(song1['key'] - song2['key'])
    energy_diff = abs(song1['energy'] - song2['energy'])
    mfcc_diff = np.linalg.norm(song1['mfcc'] - song2['mfcc'])

    quality = 1 / (1 + tempo_diff + key_diff + energy_diff + mfcc_diff)
    return quality


def merge_sort_songs(songs):
    if len(songs) <= 1:
        return songs
    mid = len(songs) // 2
    left = merge_sort_songs(songs[:mid])
    right = merge_sort_songs(songs[mid:])
    return merge(left, right)


def merge(left, right):
    sorted_songs = []
    while left and right:
        left_quality = evaluate_transition(left[0], right[0])
        right_quality = evaluate_transition(right[0], left[0])
        if left_quality > right_quality:
            sorted_songs.append(left.pop(0))
        else:
            sorted_songs.append(right.pop(0))
    sorted_songs.extend(left or right)
    return sorted_songs


def get_all_songs(directory, file_extensions=None):
    if file_extensions is None:
        file_extensions = ['.mp3', '.wav', '.flac']

    song_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in file_extensions):
                song_files.append(os.path.join(root, file))

    return song_files


def process_playlist(directory):
    file_paths = get_all_songs(directory)
    songs = [analyze_song(file_path) for file_path in file_paths]
    sorted_songs = merge_sort_songs(songs)
    return sorted_songs


if __name__ == "__main__":
    directory_path = 'songs'
    sorted_songs = process_playlist(directory_path)
    for song in sorted_songs:
        print(song['file_path'])
