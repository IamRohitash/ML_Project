import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

##data read

data=pd.read_csv('file:///C:/Users/user/Desktop/Datasets1/headbrain.csv')
data.head()

## read the data in the form of x and y
X=data['Head Size(cm^3)'].values
print(X)
Y=data['Brain Weight(grams)'].values
print(Y)



## find out mean of x and y
X_mean=np.mean(X)
print(X_mean)
Y_mean=np.mean(Y)
print(Y_mean)

## find out B1 amd b2

n=len(X)
print(n)
numer=0
demon=0

for i in range(n):
    numer+=(X[i]-X_mean)*(Y[i]-Y_mean)
    demon+=(X[i]-X_mean)**2

b1=numer/demon
print(b1)
b0=Y_mean-b1*X_mean
print(b0)

## max value of X and y

max_x=max(X)+100
min_x=min(X)-100

x=np.linspace(min_x,max_x,1000)
y=b0+b1*x

plt.plot(x,y,color='#58b970',label='LR')
plt.scatter(X,Y,color='#ef5423',label='SP')
         
## R squar method
t=0
s=0
for i in range(n):
    Ypred=b0+b1*X[i]
    t+=(Y[i]-Ypred)**2
    s+=(Y[i]-Y_mean)**2
    
r2=1-(t/s)
print(r2)    

         


    

