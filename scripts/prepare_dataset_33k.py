# %%
# https://huggingface.co/datasets/lmsys/chatbot_arena_conversations
# code for download 33k
# credits: https://www.kaggle.com/datasets/abdullahmeda/lmsys-additional-33k-labelled-conversations)

# %%
import json
import pandas as pd

from tqdm.auto import tqdm
from datasets import load_dataset
from huggingface_hub import login
from user_secrets import UserSecretsClient

# %%
tqdm.pandas()
login(UserSecretsClient().get_secret("HF_TOKEN"))

# %%
train = pd.read_csv("../ouput/train.csv")
external_data = load_dataset("lmsys/chatbot_arena_conversations")["train"].to_pandas()

print(train.columns)

# %%
def separate_conv(conv):
    try:
        user_texts = [x["content"] for x in conv if x["role"] == "user"]
        assistant_texts = [
            x["content"] for x in conv if x["role"] == "assistant"
        ]

        return user_texts, json.dumps(assistant_texts)
    except:
        print(conv)


external_data["prompt_a"], external_data["response_a"] = zip(
    *external_data.conversation_a.progress_apply(separate_conv)
)
external_data["prompt_b"], external_data["response_b"] = zip(
    *external_data.conversation_b.progress_apply(separate_conv)
)

# %%
assert (external_data["prompt_a"] == external_data["prompt_b"]).all() == True

external_data["prompt"] = external_data["prompt_a"].progress_apply(json.dumps)

# %%
external_data.winner.value_counts()


# %%
def one_hot_encode(winner):
    return pd.Series(
        [
            int("model_a" == winner),
            int("model_b" == winner),
            int("tie" == winner or "tie (bothbad)" == winner),
        ]
    )


external_data[["winner_model_a", "winner_model_b", "winner_tie"]] = (
    external_data.winner.progress_apply(one_hot_encode)
)

# %%
assert (
    external_data[["winner_model_a", "winner_model_b", "winner_tie"]]
    .sum(axis=1)
    .all()
)

# %%
external_data.columns

# %%
cols = [
    "question_id",
    "model_a",
    "model_b",
    "prompt",
    "response_a",
    "response_b",
    "winner_model_a",
    "winner_model_b",
    "winner_tie",
]

external_data = pd.DataFrame(
    external_data[cols].copy().values, columns=train.columns
)

# %%
superset = pd.concat([external_data, train]).reset_index(drop=True)
external_data_deduplicated = superset.drop_duplicates(
    subset=["prompt"], keep="last"
)
external_data_deduplicated = external_data_deduplicated[
    external_data_deduplicated.index.isin(external_data.index)
]

print(len(external_data_deduplicated))
# %%
print(external_data_deduplicated.head())

# %%
external_data.to_csv("output/lmsys-33k.csv", index=False)
external_data_deduplicated.reset_index(drop=True).to_csv(
    "output/lmsys-33k-deduplicated.csv", index=False
)

# %%
