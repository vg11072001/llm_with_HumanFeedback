import os
import json
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from kaggle.api.kaggle_api_extended import KaggleApi
from user_secrets import UserSecretsClient

secrets_client = UserSecretsClient()

os.environ['KAGGLE_USERNAME'] = secrets_client.get_secret("KAGGLE_USERNAME")
os.environ['KAGGLE_KEY'] = secrets_client.get_secret("KAGGLE_KEY")

api = KaggleApi()
api.authenticate()

competition_name = 'lmsys-chatbot-arena'
file_name = 'train.csv'
output_dir = '../input'

api.competition_download_file(competition_name, file_name, path=output_dir)

df = pd.read_csv(os.path.join(output_dir, file_name))

df.to_csv("../output/train.csv", index=False)

print('-----------train-----------')
print(df.head())
print(df.shape)
print(df.columns)
sgkf = StratifiedGroupKFold(n_splits=5, random_state=0, shuffle=True)
group_id = df["prompt"]
label_id = df["winner_model_a winner_model_b winner_tie".split()].values.argmax(1)
splits = list(sgkf.split(df, label_id, group_id))

df["fold"] = -1
for fold, (_, valid_idx) in enumerate(splits):
    df.loc[valid_idx, "fold"] = fold
print('-----------folds-----------')
print(df["fold"].value_counts())
print('-----------dtrainval-----------')
print(df.head())
print(df.shape)
print(df.columns)
df.to_csv("../output/dtrainval.csv", index=False)
