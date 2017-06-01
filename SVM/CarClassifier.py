# -*- coding: utf-8 -*-
"""
Created on Fri May 27 09:36:27 2016

@author: lgc
"""
import sys
import matplotlib.pyplot as plt   
from mpl_toolkits.mplot3d import Axes3D
from svmutil import *
import numpy as np
import random





def selectRandomJ(k, m):
    # Variable i is the index of the first alpha.  Variable m is the total number of alphas.
    # Randomly choose a an alpha value by its index and return it if it != i.
    j = k
    while j==k:
        j = int(random.uniform(0,m))
    return j


def clipAlpha(alphaJ, high, low):
    # Clip alpha value so it is within [low, high] range.
    if alphaJ > high:
        alphaJ = high

    if alphaJ < low:
        alphaJ = low

    return alphaJ

def kernel(matrix,i,j):#the gaussian kernel
    rs=0
    diff=[]
    diff=matrix[i,:]-matrix[j,:]
   
    rs=diff*diff.T
    rs =np.exp(-1*0.24*rs)
    return rs
def simpleSMO(datMatIn, classLabel, c, toler, maxIter):
    # SMO (Sequential Minimal Optimization) algorithm finds a set of alphas and the constant b
    dataMatrix = np.mat(datMatIn)
    labelMatrix = np.mat(classLabel).transpose()
    counter, b = 0, 0
    rowCnt = np.shape(dataMatrix)[0]
    alphas = np.mat(np.zeros((rowCnt,1)))

    while counter < maxIter:
        alphaPairsChanged = 0
        for i in range(rowCnt):
            # Prediction of the class. f(x) = w^T * x + b, which is also f(x) = alphas * labels * <xi, x> + b
            fOfXi = float(np.multiply(alphas,labelMatrix).T * (dataMatrix*dataMatrix[i,:].T)) + b
            # Error based on prediction and real instance of class
            errorI = fOfXi - float(labelMatrix[i])

            # If the errorI is large, the alphas corresponding to the real instance are optimized
            # Check to see is alpha equals C or 0, bc ones that are cannot be optimized
            if ((labelMatrix[i]*errorI<-toler)and(alphas[i]<c))or((labelMatrix[i]*errorI>toler)and(alphas[i]>0)):
                # Randomly select a second alpha.  Maximize alpha[i] and alpha[j] to maximize the objective function
                j = selectRandomJ(i, rowCnt)
                fOfXj = float(np.multiply(alphas,labelMatrix).T * (dataMatrix*dataMatrix[j,:].T)) + b
                # Calculate the error of the second alpha
                errorJ = fOfXj - float(labelMatrix[j])
                # old alpha [i] and [j] prior to optimization
                oldAlphaI = alphas[i].copy()
                oldAlphaJ = alphas[j].copy()

                # alpha[i] and alpha[j] are the Lagrange multipliers to optimize.  Limits are selected such that
                # low <= alpha[j] <= high to satisfy the constraint 0 <= alpha[j] <= C.
                if labelMatrix[i] != labelMatrix[j]:
                    lowLimit = max(0, alphas[j] - alphas[i])
                    highLimit = min(c, c+alphas[j] - alphas[i])
                else:
                    lowLimit = max(0, alphas[i] + alphas[j] - c)
                    highLimit = min(c, alphas[i] + alphas[j])

                # If low and high limits equal, exit loop because alpha[j] cannot be optimized
                if lowLimit == highLimit:
                    continue

                # Eta is used to calculate the optimal amount to change alpha[j]
                eta = (2.0 * kernel(dataMatrix,i,j))
                eta -= ((kernel(dataMatrix,i,i)) + (kernel(dataMatrix,j,j)))

                # If eta equals 0, exit loop because alpha[j] cannot be optimized
                if eta >= 0:
                    continue

                alphas[j] -= labelMatrix[j] * (errorI-errorJ) / eta
                # Clip alpha[j] to ensure low <= alpha[j] <= high
                alphas[j] = clipAlpha(alphas[j], highLimit, lowLimit)

                if abs(alphas[j] - oldAlphaJ) < 0.00001:
                    continue

                # alpha[i] changed in opposite direction from alpha[j]
                alphas[i] += labelMatrix[j] * labelMatrix[i] * (oldAlphaJ - alphas[j])

                # Set the constant term for this pair of optimized alphas such that the KKT conditions are satisfied
                b1 = b - errorI - ((alphas[i] - oldAlphaI) * labelMatrix[i] * kernel(dataMatrix,i,i))
                b1 -= ((alphas[j] - oldAlphaJ) * labelMatrix[j] * kernel(dataMatrix,i,j))

                b2 = b - errorJ - ((alphas[i] - oldAlphaI) * labelMatrix[i] * kernel(dataMatrix,i,j))
                b2 -= ((alphas[j] - oldAlphaJ) * labelMatrix[j] * kernel(dataMatrix,j,j))

                if 0 < alphas[i] < c:
                    b = b1
                elif 0 < alphas[j] < c:
                    b= b2
                else:
                    # If both optimized alphas are at 0/c, than all thresholds between b1 & b2 satisfy KKT conditions
                    b = (b1 + b2) / 2.0

                # No continue statements caused exit from loop so the pair of alphas are optimized
                alphaPairsChanged += 1

        # Will only exit while loop when entire date set has be traversed maxIter number of times without change.
        if alphaPairsChanged == 0:
            counter += 1
        else:
            counter = 0

    return b, alphas


def isInclassvalue(value,classvar):#check
        for i in range(len(classvar)):
            if classvar[i]==value:
                return True
        return False



def multiply(x,y):
    result=0
    for i in range(len(x)):
        result+=x[i]*y[i]
    return result
f = open('car.data','r')
data = [v.strip('\n').split(',') for v in f.readlines()]
 
samples=[]
labels=[]
features=[['vhigh','high','med','low'],['vhigh','high','med','low'],['2','3','4','5more'],['2','4','more'],['small','med','big'],['low','med','high']]
values=[[0.4,0.3,0.2,0.1],[0.4,0.3,0.2,0.1],[0.2,0.3,0.4,0.6],[0.2,0.4,0.5],[0.1,0.2,0.4],[0.1,0.2,0.4]]
#transfer the string to number
for i in range(len(data)):
    samples.append(data[i][0:len(data[0])-1])
    labels.append(data[i][len(data[0])-1])
for i in range(len(samples[0])):
    for j in range(len(samples)):
        for k in range(len(features[i])):
            if samples[j][i] == features[i][k]:
                samples[j][i] =  values[i][k]
         
         
         
         
ws=[]
bs=[]
smpcopy=[]  # save the data for libsvm 
labelcopy=[]
'''
for i in range(len(samples)):
    temp=[]
    for j in range(len(samples[0])):
        temp.append(samples[i][j])
    smpcopy.append(temp)
for i in range(len(labels)):
    labelcopy.append(labels[i])
    
    
samples=samples[0:1200]# use 1200 samples to do training
labels=labels[0:1200]
carclass=['unacc','acc','good','vgood']#4 classes

for i in range(len(carclass)):
            newclassvalue=[]
            for j in range(len(samples)):
                #take one class as positive class, and the rest as negative class
                if labels[j] == carclass[i]:
                    newclassvalue.append(1)
                else:
                    newclassvalue.append(-1)
            constantB, alphasOptimized = simpleSMO(samples, newclassvalue, 1, 0.001, 10)  
            w=[]
            # Use lagrange multipliers to calculate weight vector w
            for k in range(len(samples[0])):
                cal=0
                for j in range(len(samples)):
                    cal+=samples[j][k]* newclassvalue[j]*alphasOptimized[j]
                w.append(cal)
            # Save weight vector and threshold for prediction
            ws.append(w)
            bs.append(constantB)
            # Do prediction on training set
            prediction=[]
            for a in range(len(samples)):
                probi=multiply(w,samples[a])+constantB
                if probi > 0:
                    prediction.append(1)
                else:
                    prediction.append(-1)
            accuracy=0.0
            for b in range(len(newclassvalue)):
                if prediction[b] == newclassvalue[b]:
                    accuracy+=1
            accuracy=accuracy/(len(newclassvalue))
            print(accuracy)
# Use one-versus-rest model to do prediction on testing set           
samples=smpcopy
labels=labelcopy       
predictclass=[]
for i in range(len(samples)-1200):
    maximum=0
    predict=0
    for w in range(len(ws)):# Classify each sample into class who has the biggest decision function value
        pro=multiply(ws[w],samples[i+1200])+bs[w]
        if pro > maximum:
            maximum= pro
            predict =w
    predictclass.append(carclass[predict])

accuracy=0.0
for i in range(len(labels)-1200):
    if predictclass[i] == labels[i+1200]:
        accuracy+=1
accuracy=accuracy/(len(data)-1200)
print(accuracy)
'''




for i in range(len(samples)):
    temp=[]
    for j in range(len(samples[0])):
        temp.append(samples[i][j])
    smpcopy.append(temp)
for i in range(len(labels)):
    labelcopy.append(labels[i])
    
samples=samples[0:1200]# use 1200 samples to do training
labels=labels[0:1200]



ws=[]
bs=[]
# Do classification for each pair of 
for i in range(len(carclass)):
    for j in range(i+1,len(carclass)):
        samples1=[]# Extract samples for each SVM
        labels1=[]
        
        for k in range(len(samples)):
            if labels[k] == carclass[i]:
                samples1.append(samples[k])
                labels1.append(1)
            elif labels[k] == carclass[j]:
                samples1.append(samples[k])
                labels1.append(-1)
                
        constantB, alphasOptimized = simpleSMO(samples1, labels1, 1, 0.001, 20)  
        w=[]
        for n in range(len(samples1[0])):
            cal=0
            for m in range(len(samples1)):
                cal+=samples1[m][n]*newclassvalue[m]*alphasOptimized[m]
            w.append(cal)
        ws.append(w)
        bs.append(constantB)
        positiveclass.append(i)
        negtiveclass.append(j)


        #do prediction on training samples
        
        prediction=[]

        positive=0
        negative=0
        for h in range(len(labels1)):
            prob=multiply(w,samples1[h]) +b
            if prob > 0:
                prediction.append(1)
                positive+=1
                
            else:
                prediction.append(-1)
                negative+=1
        counter=0
        print(positive,negative)
        for l in range(len(labels1)):
            if prediction[l] == labels1[l]:
                counter+=1
        print(counter/len(labels1))
predict=[]


# do prediction on testing samples
samples=smpcopy[1200:]
labels=labelcopy[1200:]
for i in range(len(samples)):
    count=[0,0,0,0]# Count the vote for each class 
    for w in range(len(ws)):
        pro=multiply(ws[w],samples[i]) + bs[w]
        if pro > 0:
            count[positiveclass[w]]+=1
        else:
            count[negtiveclass[w]]+=1
    #It belongs to the class with maximal vote
    maxcount=-1
    maxindex=0
    for e in range(len(count)):
        if count[e] > maxcount:
            maxcount=count[e]
            maxindex=e
    predict.append(carclass[maxindex])##
accuracy2=0.0
for i in range(len(labels)):
    if predict[i] == labels[i]:
        accuracy2+=1
accuracy2=accuracy2/(len(samples))
print(accuracy2)
      
#y=[]
#for i in range(len(samples)):
#    for j in range(len(carclass)):
#        if labelcopy[i] == carclass[j]:
#            y.append(j)
#m = svm_train(y[0:1400],samples[0:1400],'-t 2 -g 0.25 -w4 2 -v 4')
#p_label, p_acc, p_val = svm_predict(y[1400:], samples[1400:], m)
#print(p_acc)

         
