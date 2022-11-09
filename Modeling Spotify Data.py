import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score



luke_df_2016 = pd.read_csv(f'C:/Users/lasmi/PyCharmProjects/SpotifyProject/Data/Luke_2016_Top_Songs.csv')
luke_df_2017 = pd.read_csv(f'C:/Users/lasmi/PyCharmProjects/SpotifyProject/Data/Luke_2017_Top_Songs.csv')
luke_df_2018 = pd.read_csv(f'C:/Users/lasmi/PyCharmProjects/SpotifyProject/Data/Luke_2018_Top_Songs.csv')
luke_df_2019 = pd.read_csv(f'C:/Users/lasmi/PyCharmProjects/SpotifyProject/Data/Luke_2019_Top_Songs.csv')
luke_df_2020 = pd.read_csv(f'C:/Users/lasmi/PyCharmProjects/SpotifyProject/Data/Luke_2020_Top_Songs.csv')

jp_df_2017 = pd.read_csv(f'C:/Users/lasmi/PycharmProjects/SpotifyProject/Data/jp_2017_Top_Songs.csv')
jp_df_2018 = pd.read_csv(f"C:/Users/lasmi/PycharmProjects/SpotifyProject/Data/jp_2018_Top_Songs.csv")
jp_df_2019 = pd.read_csv(f"C:/Users/lasmi/PycharmProjects/SpotifyProject/Data/jp_2019_Top_Songs.csv")
jp_df_2020 = pd.read_csv(f"C:/Users/lasmi/PycharmProjects/SpotifyProject/Data/jp_2020_Top_Songs.csv")

fabian_df_2018 = pd.read_csv("C:/Users/lasmi/PycharmProjects/SpotifyProject/Data/FP_2018_Top_Songs.csv")
fabian_df_2019 = pd.read_csv("C:/Users/lasmi/PycharmProjects/SpotifyProject/Data/fp_2019_Top_Songs.csv")
fabian_df_2020 = pd.read_csv("C:/Users/lasmi/PycharmProjects/SpotifyProject/Data/FP_2020_Top_Songs.csv")
fabian_df_2017 = pd.read_csv("C:/Users/lasmi/PycharmProjects/SpotifyProject/Data/FP_2017_Top_Songs.csv")

#combining each person's data
luke_df_concat = pd.concat([luke_df_2016, luke_df_2017, luke_df_2018, luke_df_2019, luke_df_2020], ignore_index=True, axis=0)
luke_df_concat['users_name'] = "Luke"

jp_df_concat = pd.concat([jp_df_2017, jp_df_2018, jp_df_2019, jp_df_2020], ignore_index=True, axis=0)
jp_df_concat['users_name'] = "JP"

fabian_df_concat = pd.concat([fabian_df_2017, fabian_df_2018, fabian_df_2019, fabian_df_2020], ignore_index=True, axis=0)
fabian_df_concat['users_name'] = "Fabian"

#Combining every data frame together into one
all_df = pd.concat([luke_df_concat, jp_df_concat, fabian_df_concat], ignore_index=True, axis=0)

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


spotify_features = ['energy',
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

X = all_df[spotify_features]
y = all_df['users_name']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

#multiclass is exclusive and multilabel is not

#Setting up a pipeline that scales and then utilizes multinomical logistic regression
pipe = Pipeline([('scaler', StandardScaler()),     # Step 1
                 ('model', LogisticRegression(multi_class='multinomial', solver='lbfgs')) # Step 2
                 ])
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, pred))


#Preforming cross-validation

cv = KFold(n_splits=10, random_state=123, shuffle=True)

scores = cross_val_score(pipe, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
scores

#More metrics

accuracy = accuracy_score(y_test, pred)
print(f"Accuracy: {accuracy:.2f}")

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import classification_report
print(classification_report(y_test, pred))

#Setting up a confusion 3x3 confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, pred)

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc, average_precision_score


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

