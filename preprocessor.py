import json
import pickle
import pandas as pd

def load_encoder_state(filepath="encoder_state.pkl"):
    with open(filepath, "rb") as file:
        encoder_state = pickle.load(file)
        return encoder_state
    
def encode_input_data(input_json, encoder_state):
    # Convert the input JSON to a DataFrame
    input_df = pd.DataFrame([input_json])

    one_hot_encoded_dfs = []  # To hold one-hot encoded DataFrames for each column before concatenating
    for column, top_entities in encoder_state.items():
        # Prepare a dict to hold the encoded data
        encoded_data = {f'{column}_is_{entity}': [] for entity in top_entities}

        # Process each row in the DataFrame (although there's likely only one row)
        for index, row in input_df.iterrows():
            entities = [row[column]] if isinstance(row[column], str) else row[column]
            if entities is None:
                for entity in top_entities:
                    encoded_data[f'{column}_is_{entity}'].append(0)
                continue
            for entity in top_entities:
                encoded_data[f'{column}_is_{entity}'].append(1 if entity in entities else 0)

        # Convert the encoded data to a DataFrame
        encoded_df = pd.DataFrame(encoded_data, index=input_df.index)
        one_hot_encoded_dfs.append(encoded_df)

    # Concatenate all one-hot encoded DataFrames with the original DataFrame
    input_df = pd.concat([input_df] + one_hot_encoded_dfs, axis=1)

    return input_df
    
def preprocess_input(input_data, encoder_state):
    data = encode_input_data(input_data, encoder_state)

    columns_to_drop = ['genres', 'production_companies', 'spoken_languages',
                 'director_name', 'country']
    data = data.drop(columns=columns_to_drop)
    data = data.fillna(0)
    data = data.reset_index()

    taglines = data["tagline"].replace(0, "")
    overview = data["overview"].replace(0, "")

    descriptions = taglines + " " + overview
    descriptions = descriptions.map(lambda x: x.strip() if isinstance(x, str) else x)

    X = data.drop(columns=['overview', 'tagline'], axis=1)
    
    return (X, descriptions)


if __name__ == "__main__":
    with open("fake_data.json", "r") as file:
        data = json.load(file)

    encoder_state = load_encoder_state()
    (X, descriptions) = preprocess_input(data, encoder_state)

