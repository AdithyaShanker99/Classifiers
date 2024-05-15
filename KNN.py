import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# optimize using ball tree
# optimize using kD tree

def load_data(url) :
    df = pd.read_csv(url)
    return df

def process_data(df) :
    df['variety'] = df['variety'].replace('Setosa', 0.0)
    df['variety'] = df['variety'].replace('Versicolor', 1.0)
    df['variety'] = df['variety'].replace('Virginica', 2.0)
    features = df.drop('variety', axis=1)
    output = df['variety']
    return features, output

def getTrainAndTest(features, output) :
    X_train, X_test, Y_train, Y_test = train_test_split(features, output, test_size=0.2, random_state=41)
    return X_train, X_test, Y_train, Y_test


def find_distances(X_train, data, metric) :
    distances = []
    if metric == 'manhattan' :
        for index, train_instance in X_train.iterrows():
            distance = np.diff(train_instance - data)
            #print(f'index: {index}\ntrain_instance: {train_instance}\n\n\n\n')
            distances.append((distance, index))
    else :
        for index, train_instance in X_train.iterrows():
            distance = np.sqrt(np.sum((train_instance - data) ** 2))
            #print(f'index: {index}\ntrain_instance: {train_instance}\n\n\n\n')
            distances.append((distance, index))
    return distances


def predict(k, distances, Y_train):
    # Sort distances and get the top k elements
    sorted_list = sorted(distances)[:k]
    
    # Basic voting: Collecting votes
    labels = [Y_train.loc[index] for _, index in sorted_list]
    vote_counts = np.bincount(labels)
    
    # Check for tie
    max_votes = np.max(vote_counts)
    candidates = np.where(vote_counts == max_votes)[0]

    if len(candidates) == 1:
        # No tie, return the class with the most votes
        return candidates[0]
    else:
        # Tie exists, resolve it using weights
        weights = {candidate: 0 for candidate in candidates}
        for dist, index in sorted_list:
            label = Y_train.loc[index]
            if label in weights:
                if dist == 0:
                    weights[label] += 1e9  # Very large weight for zero distance
                else:
                    weights[label] += 1 / dist  # Weight is inverse of distance
        
        # Return the label of the candidate with the highest weight
        return max(weights, key=weights.get)


df = load_data('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
features, output = process_data(df)
X_train, X_test, Y_train, Y_test = getTrainAndTest(features, output)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(accuracy)



correct = 0
for index, data in X_test.iterrows() :
    distances = find_distances(X_train, data, "euclidean")
    output = predict(3, distances, Y_train)
    #print(f'Predicted output: {output}  Expected output: {Y_test.loc[index]}')
    if output == Y_test.loc[index]:
        correct+=1
print(correct/len(X_test))