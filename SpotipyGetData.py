import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import numpy as np


#File contains Client_ID and Client_Secret_ID required to use Spotify's API
spotify_client_info = pd.read_csv("C:\\Users\\lasmi\\PycharmProjects\\SpotifyProject\\Data\\spotify_client_info.csv")

client_id = spotify_client_info.iloc[0,0]
client_secret = spotify_client_info.iloc[0,1]

client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

#Function to get tracks from a specific user's playlist
def getTrackIDs(user, playlist_id):
    ids = []
    playlist = sp.user_playlist(user, playlist_id)
    for item in playlist['tracks']['items']:
        track = item['track']
        ids.append(track['id'])
    return ids


#Gets Spotify's song features from the song's ID
def getTrackFeatures(id):
    meta = sp.track(id)
    features = sp.audio_features(id)

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

#Loops over track ids to get all songs in playlist
def loop_playist(playlist_ids):
    tracks = []
    for i in range(len(playlist_ids)):
        time.sleep(.2)
        track = getTrackFeatures(playlist_ids[i])
        tracks.append(track)
    return tracks

#Creates dataframes with columns for each of Spotify's features
def get_spotify_df(tracks, year):
    df = pd.DataFrame(tracks, columns = ['name', 'album', 'artist', 'release_date',
                                         'length', 'popularity', 'acousticness', 'danceability',
                                         'energy', 'instrumentalness', 'liveness', 'loudness',
                                         'speechiness', 'tempo', 'valence', 'time_signature',
                                         'key', 'mode', 'uri'])
    return df


#Add release year to the dataframe
def get_years(df):
    years = []
    for date in df['release_date'].values:
        if '-' in date:
            years.append(date.split('-')[0])
        else:
            years.append(date)
    df['release_year'] = years
    return df

#CSV that includes the username and playlist ID for each year between 2016-2020
spotify_users_and_playlists = pd.read_csv("C:\\Users\\lasmi\\PycharmProjects\\SpotifyProject\\Data\\spotify_users_and_playlists.csv")

luke_user = spotify_users_and_playlists.iloc[0,0]

luke_playlist_2016 = spotify_users_and_playlists.iloc[0,1]
luke_playlist_2017 = spotify_users_and_playlists.iloc[1,1]
luke_playlist_2018 = spotify_users_and_playlists.iloc[2,1]
luke_playlist_2019 = spotify_users_and_playlists.iloc[3,1]
luke_playlist_2020 = spotify_users_and_playlists.iloc[4,1]

ids_2016 = getTrackIDs(luke_user, luke_playlist_2016)
ids_2017 = getTrackIDs(luke_user, luke_playlist_2017)
ids_2018 = getTrackIDs(luke_user, luke_playlist_2018)
ids_2019 = getTrackIDs(luke_user, luke_playlist_2019)
ids_2020 = getTrackIDs(luke_user, luke_playlist_2020)



time_start = time.time()
ids_2016_playlist_loop = loop_playist(ids_2016)
ids_2017_playlist_loop = loop_playist(ids_2017)
ids_2018_playlist_loop = loop_playist(ids_2018)
ids_2019_playlist_loop = loop_playist(ids_2019)
ids_2020_playlist_loop = loop_playist(ids_2020)
time_end = time.time()
print((time_end - time_start)/60)

df_2016 = get_spotify_df(ids_2016_playlist_loop, 2016)
df_2017 = get_spotify_df(ids_2017_playlist_loop, 2017)
df_2018 = get_spotify_df(ids_2018_playlist_loop, 2018)
df_2019 = get_spotify_df(ids_2019_playlist_loop, 2019)
df_2020 = get_spotify_df(ids_2020_playlist_loop, 2020)

# save dataframes
dfs = [df_2016,df_2017,df_2018,df_2019,df_2020]
names = ['Luke_2016', 'Luke_2017', 'Luke_2018', 'Luke_2019', 'Luke_2020']

for df, name in zip(dfs, names):
    df.to_csv(f'C:/Users/lasmi/PyCharmProjects/SpotifyProject/Data/{name}_Top_Songs.csv', index=False)


df_2016 = pd.read_csv(f'C:/Users/lasmi/PyCharmProjects/SpotifyProject/Data/Luke_2016_Top_Songs.csv')
df_2017 = pd.read_csv(f'C:/Users/lasmi/PyCharmProjects/SpotifyProject/Data/Luke_2017_Top_Songs.csv')
df_2018 = pd.read_csv(f'C:/Users/lasmi/PyCharmProjects/SpotifyProject/Data/Luke_2018_Top_Songs.csv')
df_2019 = pd.read_csv(f'C:/Users/lasmi/PyCharmProjects/SpotifyProject/Data/Luke_2019_Top_Songs.csv')
df_2020 = pd.read_csv(f'C:/Users/lasmi/PyCharmProjects/SpotifyProject/Data/Luke_2020_Top_Songs.csv')


