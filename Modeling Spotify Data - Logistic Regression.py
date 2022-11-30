#Importing and structuring our data for analysis

#Importing necessary packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


#Bringing in all of the individual data sets for each person's playlists by year
luke_df_2016 = pd.read_csv(f'Data/Luke_2016_Top_Songs.csv')
luke_df_2017 = pd.read_csv(f'Data/Luke_2017_Top_Songs.csv')
luke_df_2018 = pd.read_csv(f'Data/Luke_2018_Top_Songs.csv')
luke_df_2019 = pd.read_csv(f'Data/Luke_2019_Top_Songs.csv')
luke_df_2020 = pd.read_csv(f'Data/Luke_2020_Top_Songs.csv')

jp_df_2017 = pd.read_csv(f'Data/jp_2017_Top_Songs.csv')
jp_df_2018 = pd.read_csv(f"Data/jp_2018_Top_Songs.csv")
jp_df_2019 = pd.read_csv(f"Data/jp_2019_Top_Songs.csv")
jp_df_2020 = pd.read_csv(f"Data/jp_2020_Top_Songs.csv")

fabian_df_2018 = pd.read_csv(f"Data/FP_2018_Top_Songs.csv")
fabian_df_2019 = pd.read_csv(f"Data/fp_2019_Top_Songs.csv")
fabian_df_2020 = pd.read_csv(f"Data/FP_2020_Top_Songs.csv")
fabian_df_2017 = pd.read_csv(f"Data/FP_2017_Top_Songs.csv")

#Creating a function to add a new variable to the dataframes, the song release year
def get_years(df):
    years = []
    for date in df['release_date'].values:
        if '-' in date:
            years.append(date.split('-')[0])
        else:
            years.append(date)
    df['release_year'] = years
    return df


#Adding release year to each dataframe for each individual
luke_df_2016 = get_years(luke_df_2016)
luke_df_2017 = get_years(luke_df_2017)
luke_df_2018 = get_years(luke_df_2018)
luke_df_2019 = get_years(luke_df_2019)
luke_df_2020 = get_years(luke_df_2020)

jp_df_2017 = get_years(jp_df_2017)
jp_df_2018 = get_years(jp_df_2018)
jp_df_2019 = get_years(jp_df_2019)
jp_df_2020 = get_years(jp_df_2020)

fabian_df_2018 = get_years(fabian_df_2018)
fabian_df_2019 = get_years(fabian_df_2019)
fabian_df_2020 = get_years(fabian_df_2020)
fabian_df_2017 = get_years(fabian_df_2017)


#Adding a new variable, "users_name", and combining each person's data
luke_df_concat = pd.concat([luke_df_2016, luke_df_2017, luke_df_2018, luke_df_2019, luke_df_2020], ignore_index=True, axis=0)
luke_df_concat['users_name'] = "Luke"

jp_df_concat = pd.concat([jp_df_2017, jp_df_2018, jp_df_2019, jp_df_2020], ignore_index=True, axis=0)
jp_df_concat['users_name'] = "JP"

fabian_df_concat = pd.concat([fabian_df_2017, fabian_df_2018, fabian_df_2019, fabian_df_2020], ignore_index=True, axis=0)
fabian_df_concat['users_name'] = "Fabian"

#Combining every data frame together into a final one
all_df = pd.concat([luke_df_concat, jp_df_concat, fabian_df_concat], ignore_index=True, axis=0)
all_df["release_year"] = pd.to_numeric(all_df["release_year"])



# remove repeats on individual playlists
def remove_repeats(df):
    rows_old = range(len(df['name']))
    rows_new = []
    skips = []
    for i in range(len(df['name'])):
        for j in range(i+1, len(df['name'])):
            if (df['name'][i] == df['name'][j]) and (df['artist'][i] == df['artist'][j]):
                skips.append(j)
    for row in rows_old:
        if not row in skips:
            rows_new.append(row)
    df = df.iloc[rows_new,:].reset_index(drop=True)
    return df

remove_repeats(all_df)


# Create function to do linear transformation on variable to change value to [0,1]
def convert_scale(df, col):
    df[col + '_old'] = df[col]
    new_max = 1
    new_min = 0
    new_range = new_max-new_min
    max_val = df[col].max()
    min_val=df[col].min()
    val_range = max_val - min_val
    df[col]=df[col].apply(lambda x: (((x-min_val)*new_range)/val_range)+new_min)
    return

#Setting the numerical spotify features
numeric_spotify_features = ['energy',
    'valence',
    'danceability',
    'liveness',
    'speechiness',
    'instrumentalness',
    'acousticness',
    'loudness',
    'length',
    'popularity',
    'tempo',]

for col in numeric_spotify_features:
    convert_scale(all_df, col)

#Now we need to address the categorical variables. For this, we will utilize One Hot Encoder
ohe = OneHotEncoder()
ohe_results = ohe.fit_transform(all_df[['time_signature', 'mode', 'key']])
onehot_df = pd.DataFrame(ohe_results.toarray(), columns=['time_signature_1', 'time_signature_2',
                                                     'time_signature_3', 'time_signature_4',
                                                     'mode_1', 'mode_2',
                                                     'key_1', 'key_2', 'key_3', 'key_4', 'key_5', 'key_6',
                                                     'key_7', 'key_8', 'key_9', 'key_10', 'key_11', 'key_12'])
all_df = pd.concat([all_df, onehot_df], axis=1)

#Finally, we combine all of our features used for prediction into one list
all_spotify_features = ['energy',
    'valence',
    'danceability',
    'liveness',
    'speechiness',
    'instrumentalness',
    'acousticness',
    'loudness',
    'length',
    'popularity',
    'tempo',
    'release_year',
    'time_signature_1',
    'time_signature_2',
    'time_signature_3',
    'time_signature_4',
    'mode_1',
    'mode_2',
    'key_1',
    'key_2',
    'key_3',
    'key_4',
    'key_5',
    'key_6',
    'key_7',
    'key_8',
    'key_9',
    'key_10',
    'key_11',
    'key_12']

#Now, we can finally set our X and y values. X will be all the Spotify features provided above and y will be each individual's name
X = all_df[all_spotify_features]
y = all_df['users_name']

#Here we split our data up into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)


#Setting up a pipeline that scales and then utilizes multinomial logistic regression
pipe = Pipeline([('scaler', StandardScaler()),     # Step 1
                 ('model', LogisticRegression(multi_class='multinomial', solver='lbfgs')) # Step 2
                 ])
#Fitting and predicting with our data
pipe.fit(X_train, y_train)
pred_logi = pipe.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, pred_logi))


#Preforming cross-validation
cv = KFold(n_splits=10, random_state=123, shuffle=True)
scores = cross_val_score(pipe, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
scores

#More evalutation metrics including accuracy score and a classification report
accuracy = accuracy_score(y_test, pred_logi)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, pred_logi))


#Finding the F1 scores
from sklearn.metrics import f1_score


#F1 score is always between 0 and 1
#Score of 0 is bad, score of 1 is good

f1_logi = f1_score(y_test, pred_logi, average='macro')
print("F1 Logistic (base):",f1_logi)

print("\n  The difference between macro, micro, and weighted is this:")
print("  basically, since we have a 3x3 confusion matrix, we don't only have")
print("  ONE kind of false positive, true positive, etc...")
print("  the model can predict a true jp as being luke OR fabio, meaning our TN, TP, FN, FP")
print("  are not so straightforward")
print("\n  --macro, micro, and weighted are just different ways of combining the 3x3 matrix")
print("  to get a better picture of your f1 score")

#Calculating our Matthews Correlation Coefficient
from sklearn.metrics import matthews_corrcoef

mcc_logi = matthews_corrcoef(y_test, pred_logi)
print("MCC Logistic (base):",mcc_logi)
