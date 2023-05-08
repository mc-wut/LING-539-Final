"""
NAME = "Matthew McLaughlin"
# University of Arizona email address
EMAIL = "mclaughlinm@arizona.edu"
# Names of any collaborators.  Write N/A if none.
COLLABORATORS = "N/A"
"""


from typing import Iterator, Iterable, List, Tuple, Text, Union 
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import spmatrix
NDArray = Union[np.ndarray, spmatrix]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import re
import csv
import unicodedata


"""
After a decent amount of modifications and alternate settings, I arrived
at the conclusion that my most basic attempt was the most functional.
1,2 NGrams and CountVectorizer with liblinear regression.
""" 


def read_training(file_path: Text) -> Iterator[Tuple[Text, Text, Text]]:
    """
    Generates id/text/label tupes from the training file.

    """
    training_tuples = []

    f = open(file_path)

    data = pd.read_csv(f)

    return data
 
data = read_training("./train.csv")
text = data['TEXT'].astype('U').values
training_labels = data['LABEL'].astype('U').values

dev = read_training("./test.csv")
dev_text = dev['TEXT'].astype('U').values
dev_ids = dev['ID'].astype('U').values



vectorizer = CountVectorizer(decode_error='ignore', ngram_range=(1,2))
vectorizer.fit_transform(text)



lbl_encoder = LabelEncoder()
lbl_encoder.fit_transform(training_labels)

clf = LogisticRegression(solver='liblinear', penalty='l2',
                          class_weight='balanced')

clf.fit(vectorizer.transform(text), lbl_encoder.transform(training_labels))

predicted_indices = clf.predict(vectorizer.transform(dev_text))


zip = np.column_stack((dev_ids,predicted_indices))
df = pd.DataFrame(zip, columns= ["ID", "LABEL"])

fp = Path('./submission.csv')
df.to_csv(fp,index=False)