#This will get the URL for an artist image given the artist’s name
import spotipy
import sys
import time
import numpy as np
import pandas as pd

from spotipy.oauth2 import SpotifyClientCredentials
spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

#first bit of data

df_2020 = pd.read_csv('your_top_songs_2020.csv')

#Inspect data

#df_2020.head()

# my spotify username and playlist ids
# on playlist page, click on "..." -> then on "Share" -> then "Copy Spotify URI"
def getTrackIDs(user, playlist_id):
    ids = []
    playlist = spotify.user_playlist(user, playlist_id)
    for item in playlist['tracks']['items']:
        track = item['track']
        ids.append(track['id'])
    return ids



def getTrackFeatures(id):
    meta = spotify.track(id)
    features = spotify.audio_features(id)

    # meta
    name = meta['name']
    album = meta['album']['name']
    artist = meta['album']['artists'][0]['name']
    release_date = meta['album']['release_date']
    length = meta['duration_ms']
    popularity = meta['popularity']

    # features
    acousticness = features[0]['acousticness']
    danceability = features[0]['danceability']
    energy = features[0]['energy']
    instrumentalness = features[0]['instrumentalness']
    liveness = features[0]['liveness']
    loudness = features[0]['loudness']
    speechiness = features[0]['speechiness']
    tempo = features[0]['tempo']
    valence = features[0]['valence']
    time_signature = features[0]['time_signature']
    key = features[0]['key']
    mode = features[0]['mode']
    uri = features[0]['uri']

    track = [name, album, artist, release_date,
             length, popularity, acousticness,
             danceability, energy, instrumentalness,
             liveness, loudness, speechiness, tempo,
             valence, time_signature,
             key, mode, uri]
    return track

# loop over track ids to get all songs in playlist
def loop_playlist(playlist_ids):
    tracks = []
    for i in range(len(playlist_ids)):
        time.sleep(.2)
        track = getTrackFeatures(playlist_ids[i])
        tracks.append(track)
    return tracks

loop_playlist('37i9dQZF1ELZEx9G8j0DrH')


def get_spotify_df(tracks, year):
    df = pd.DataFrame(tracks, columns = ['name', 'album', 'artist', 'release_date',
                                         'length', 'popularity', 'acousticness', 'danceability',
                                         'energy', 'instrumentalness', 'liveness', 'loudness',
                                         'speechiness', 'tempo', 'valence', 'time_signature',
                                         'key', 'mode', 'uri'])
    return df

def get_years(df):
    years = []
    for date in df['release_date'].values:
        if '-' in date:
            years.append(date.split('-')[0])
        else:
            years.append(date)
    df['release_year'] = years
    return df















