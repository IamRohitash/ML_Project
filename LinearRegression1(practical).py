import numpy as np
import matplotlib.pyplot as plt

def estimate_coef(x,y):
    n=np.size(x)
    m_x,m_y=np.mean(x),np.mean(y)
    
    SS_xx=np.sum(x*x)-n*m_x*m_x
    SS_xy=np.sum(x*y)-n*m_x*m_y
    
    b_1=SS_xy/SS_xx
    b_0=m_y-b_1*m_x
    
    return(b_0,b_1)
    
def plot_regression_line(x,y,b):
    
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30) 
    y_pred=b[0]+b[1]*x
    
    plt.plot(x,y_pred,color='g')
    plt.xlabel('X(feature)')
    plt.ylabel('Y(response)')
    plt.show()
  
    
    
    
def main():
    # observations 
    x=np.array([0,1,2,3,4,5,6,7,8,9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12]) 
    
    #estimating efficient
    b = estimate_coef(x, y)
    print('estimeted cofficient:\nb_0={}\nb_1={}'.format(b[0],b[1]))
    plot_regression_line(x,y,b)
    
    
if __name__=="__main__":
    main()