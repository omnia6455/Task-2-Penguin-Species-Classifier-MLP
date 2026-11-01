import pandas as pd
import numpy as np

def preproccessData(filepath='penguins.csv', seed=42):
    data = pd.read_csv(filepath)
    if 'OriginLocation' in data.columns:
        data['OriginLocation'] = data['OriginLocation'].map({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})

    features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'OriginLocation', 'BodyMass']
    classes = ['Adelie', 'Chinstrap', 'Gentoo']
    target_map = {'Adelie':[1, 0, 0], 'Chinstrap':[0, 1, 0],'Gentoo':[0, 0, 1] }

    train_list, test_list = [], []
    for cls in classes:
        cls_data = data[data['Species'] == cls].copy()
        train_part = cls_data.sample(n=30, random_state=seed)
        test_part = cls_data.drop(train_part.index).sample(n=20, random_state=seed)

        train_list.append(train_part)
        test_list.append(test_part)

    train_df = pd.concat(train_list).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_list).sample(frac=1, random_state=seed).reset_index(drop=True)

    for col in features:
        if col == 'OriginLocation':
            mode_val = train_df[col].mode()[0]
            train_df[col] = train_df[col].fillna(mode_val)
            test_df[col]  = test_df[col].fillna(mode_val)
        else:
            mean_val = train_df[col].mean()
            train_df[col] = train_df[col].fillna(mean_val)
            test_df[col]  = test_df[col].fillna(mean_val)

    X_train = train_df[features].to_numpy(dtype=float)
    X_test = test_df[features].to_numpy(dtype=float)

    train_mean = X_train.mean(axis=0)
    train_std = np.where(X_train.std(axis=0) == 0, 1.0, X_train.std(axis=0))

    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    y_train = np.array([target_map[species] for species in train_df['Species']])
    y_test = np.array([target_map[species] for species in test_df['Species']])

    return X_train, y_train, X_test, y_test
