#importing libraries
import pandas as pd

#loading data
data = pd.read_csv("bbc-text.csv")

#data pre-processing
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
stop_words = stopwords.words('english')
X = []
for i in range(2225):
    text = re.sub('[^a-zA-Z]',' ',data['text'][i])
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in stop_words]
    text = ' '.join(text)
    X.append(text)
    
#storing categories in y
y= []
yval={"sport":0, "business":1, "politics":2, "tech":3, "entertainment":4}
for j in range (2225):
    if data['category'][j] in yval:
        y.append(yval[data['category'][j]])

#creating training and testing variables
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer()
tv.fit(X_train)
X_train = tv.transform(X_train).toarray()

#fitting logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model = lr.fit(X_train, y_train)

#transforming X_test
X_test = tv.transform(X_test).toarray()

#predicting the test set
prediction = lr.predict(X_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)

#printing values
from sklearn.metrics import accuracy_score, classification_report 
print("Confusion Matrix: \n",cm)
print('Accuracy:',accuracy_score(y_test, prediction))
print('Report: \n',classification_report(y_test, prediction))