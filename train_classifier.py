#train classifier
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data_dict = pickle.load(open('data_pickle','rb'))
data = data_dict['data']
labels = data_dict['labels']

correct_length = 42

filtered_data = []
filtered_labels = []

for i, sample in enumerate(data):
    if len(sample)== correct_length:
        filtered_data.append(sample)
        filtered_labels.append(labels[i])
    else:
        print(f"Removed sample {i} with length {len(sample)}")

print(f"Original smaples:{len(data)}")
print(f"Filtered samples:{len(filtered_data)}")
print(f"Removed samples: {len(data)-len(filtered_data)}")


data_array = np.asarray(filtered_data)
labels_array = np.asarray(filtered_labels)
 
X_train,X_test,y_train,y_test = train_test_split(data_array,labels_array,test_size=0.2, shuffle=True, stratify=labels_array)

model = RandomForestClassifier()


model.fit(X_train,y_train)
y_predict = model.predict(X_test)

score = accuracy_score(y_test,y_predict)
print(f'{score *100:.2f}% of samples were classified correctly')

f = open("asl_alphabet_model.p", 'wb')
pickle.dump({'model':model},f)
f.close()