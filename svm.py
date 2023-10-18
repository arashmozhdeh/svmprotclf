import pandas as pd
import numpy as np
import argparse
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def get_3grams(seq):
    return [seq[i:i+3] for i in range(len(seq) - 2)]

def sequence_to_vector(seq, protvec_dict):
    grams = get_3grams(seq)
    vectors = [protvec_dict[g] if g in protvec_dict else [0]*100 for g in grams]
    return np.sum(vectors, axis=0)

def load_protvec(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = [line.strip('"').split('\t') for line in lines]
    return {item[0]: [float(val.replace('"', '')) for val in item[1:]] for item in data}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Protein classification')
    parser.add_argument('--shot_size', type=int, default=None, help='Number of samples per class for training. If None, use the whole dataset.')
    parser.add_argument("--train_path", default="./dataset/train_dataset.csv", type=str, help="Path to training dataset")
    parser.add_argument("--test_path", default="./dataset/test_dataset.csv", type=str, help="Path to test dataset")
    parser.add_argument('--protvec_path', default="./dataset/protVec_100d_3grams.csv", type=str, help='Path to the protVec embeddings')

    args = parser.parse_args()

    protvec_dict = load_protvec(args.protvec_path)

    train_dataset = pd.read_csv(args.train_path)
    test_dataset = pd.read_csv(args.test_path)

    # If shot_size is provided, sample based on shot size. Otherwise, use the whole dataset.
    if args.shot_size:
        train_dataset = train_dataset.groupby('Protein families').apply(lambda x: x.sample(min(len(x), args.shot_size))).reset_index(drop=True)

    # Convert protein families to numbers
    le = LabelEncoder()
    train_dataset['Protein families'] = le.fit_transform(train_dataset['Protein families'])
    test_dataset['Protein families'] = le.transform(test_dataset['Protein families'])

    train_dataset['Vector'] = train_dataset['Sequence'].apply(lambda x: sequence_to_vector(x, protvec_dict))
    test_dataset['Vector'] = test_dataset['Sequence'].apply(lambda x: sequence_to_vector(x, protvec_dict))

    X_train = np.array(train_dataset['Vector'].tolist())
    y_train = train_dataset['Protein families']

    X_test = np.array(test_dataset['Vector'].tolist())
    y_test = test_dataset['Protein families']

    # SVM training
    clf = SVC()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy on test data: {accuracy:.4f}")
