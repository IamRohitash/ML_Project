import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 

col_names=[' Rohitash','  Mohit','  Yadav','  Kumar','  Hello']
balance_data = pd.read_csv( 'file:///C:/Users/sa/Desktop/balance-scale.data', sep= ',', header = None,names=col_names) 
print ("Dataset Length: ", len(balance_data)) 
print ("Dataset Shape: ", balance_data.shape) 
print (balance_data.head()) 
    
##seperated the targeted value
X=balance_data.values[:,1:5]
Y=balance_data.values[:,0]
print(X)
print(Y)

#spliting dataset into test and training
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.3,random_state=100)
entropy=DecisionTreeClassifier(criterion='entropy',random_state=100,
                               max_depth=3,min_samples_leaf=5)
entropy.fit(X_train,Y_train)

Y_pred_en=entropy.predict(X_test)
print(Y_pred_en)

##check accuracy
print('accuracy:', accuracy_score(Y_test,Y_pred_en)*100)
print('report:',classification_report(Y_test,Y_pred_en))
print('confusion_matrix:\n',confusion_matrix (Y_test,Y_pred_en))

