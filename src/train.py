import pandas as pd
import numpy as np
import ktrain
from ktrain import text
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

PATH_DATA = '../data/'
#PATH_MODELS = '../models/'
LABELS = [0, 1]
MODEL_NAME = 'bert-base-uncased'
MAXLEN = 150

df = pd.read_csv(PATH_DATA + 'C3.csv')

# Stratified train/test splitting
ss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, intermediate_index in ss1.split(df, df['constructive_binary']):
    intermediate_set = df.loc[train_index]
    test_set = df.loc[intermediate_index]

# Indexes are unordered, put them back
intermediate_set.reset_index(drop=True, inplace=True)
test_set.reset_index(drop=True, inplace=True)

# Stratified train/val splitting
ss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
for train_index, val_index in ss2.split(intermediate_set, intermediate_set['constructive_binary']):
    train_set = intermediate_set.loc[train_index]
    val_set = intermediate_set.loc[val_index]

# Indexes are unordered, put them back
train_set.reset_index(drop=True, inplace=True)
val_set.reset_index(drop=True, inplace=True)

# Inputs
X_train = np.asarray(train_set.comment_text)
X_val = np.asarray(val_set.comment_text)
X_test = np.asarray(test_set.comment_text)

# Outputs
y_train = train_set.constructive_binary.astype('int8')
y_val = val_set.constructive_binary.astype('int8')
y_test = test_set.constructive_binary.astype('int8')

# Loading transformer model
t = text.Transformer(MODEL_NAME, maxlen=MAXLEN, class_names=LABELS)

# Preprocessing step
train_final = t.preprocess_train(X_train, y_train)
val_final = t.preprocess_test(X_val, y_val)

# Model generation
model = t.get_classifier()

# Learner object generation
learner = ktrain.get_learner(model, train_data=train_final, val_data=val_final, batch_size=32)

# Fitting the training data
learner.fit_onecycle(1e-5, 4)

# Validation step
learner.validate(class_names=t.get_classes())

# Get predictor object
predictor = ktrain.get_predictor(learner.model, preproc=t)

# Test step
y_pred = predictor.predict(X_test)

# Results visualization
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred, target_names=['Not constructive', 'Constructive']))
print(confusion_matrix(y_test, y_pred))
