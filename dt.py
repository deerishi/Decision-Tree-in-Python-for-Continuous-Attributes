import numpy as np
from copy import copy
class DecisionTree:
    
    def __init__(self):
        self.train=np.array([])
        self.labels=[]
        self.labelsT=[]
        self.Test=np.array([])
        self.usedThresholds={}
        for i in range(0,16):
            self.usedThresholds[i]=set()
        self.Tree=np.ones((1000,1))
        self.Thresholds=np.ones((1000,1))
        self.decisions={}
        self.Tree=-1*self.Tree
        self.last=0
        pass
    
    def findIG(self,data1,threshold,Attr,labels1):
        data=copy(data1)
        labels=copy(labels1)
        
        rowsLeft=np.where(data[:,Attr]>=threshold)[0]
        rowsRight=np.where(data[:,Attr]<threshold)[0]
        
        #Calculate parent threshold 
        rowsH=np.where(labels=='healthy')[0]
        rowsC=np.where(labels=='colic')[0]
        pH=float(rowsH.shape[0])/labels.shape[0]
        pC=float(rowsC.shape[0])/labels.shape[0]
        
        if pH==0 or pC==0:
            HX=0
        else:
            HX=-1*pH*np.log2(pH) - pC*np.log2(pC)
        
        #now calculate the H(Y|X)
        #print 'in IG labels.shape is ',labels.shape
        
        
        labelsLeft=copy(labels[rowsLeft])
        labelsRight=copy(labels[rowsRight])
        #print 'in IG labelsLeft.shape is ',labelsLeft.shape
        #print 'in IG labelsRight.shape is ',labelsRight.shape
        #For Left Child
        rowsH=np.where(labelsLeft=='healthy')[0]
        rowsC=np.where(labelsLeft=='colic')[0]
        
        pHL=float(rowsH.shape[0])/labelsLeft.shape[0]
        pCL=float(rowsC.shape[0])/labelsLeft.shape[0]
        if pHL==0 or pCL==0:
            HY_X_L=0
        else:
            HY_X_L=-1*pHL*np.log2(pHL) - pCL*np.log2(pCL)
            HY_X_L=HY_X_L*float(rowsLeft.shape[0])/data.shape[0]
        
        
        #For Right Child
        rowsH=np.where(labelsRight=='healthy')[0]
        rowsC=np.where(labelsRight=='colic')[0]
        
        #print 'labelsRight.shape[0] is ',labelsRight.shape[0]
        pHR=float(rowsH.shape[0])/labelsRight.shape[0]
        pCR=float(rowsC.shape[0])/labelsRight.shape[0]
        
        if pHR==0 or pCR==0:
            HY_X_R=0
        else:
            HY_X_R=-1*pHR*np.log2(pHR) - pCR*np.log2(pCR)
            HY_X_R=HY_X_R*float(rowsRight.shape[0])/data.shape[0]
        
        IG=HX-HY_X_L-HY_X_R
        return IG        
        
        
       
    def findThresholdAndIG(self,data1,Attr,labels1):
        #print 'trying attribute ',Attr
        data=copy(data1)
        labels=copy(labels1)
        values=set(data[:,Attr])
        values=copy(sorted(values))
        toTryThreshholds=[]
        for i in range(0,len(values)-1):
           toTryThreshholds.append((values[i]+values[i+1])/2)
        
        toTryThreshholds=set(toTryThreshholds)
        #print 'toTryThreshholds is ',toTryThreshholds
        if Attr in self.usedThresholds:
            for used in self.usedThresholds[Attr]:
                if used in toTryThreshholds:
                    toTryThreshholds.remove(used)
        
        #now we have all the thresholds that we need to try
        toTryThreshholds=copy(sorted(toTryThreshholds))
        IG=[]
        
        
        for threshold in toTryThreshholds:
            IG.append(self.findIG(data,threshold,Attr,labels))
        
        maxIG=max(IG)
        maxThresh=IG.index(maxIG)
        
        return toTryThreshholds[maxThresh],maxIG
            
        
                                
        
    def contructTree(self,data1,nodeNum,labels1):
            #since its a recursive function we need to have a base case. return when the number of wrong classes is 0. maybe 
            #we can chane it later
            print 'nodeNum is ',nodeNum
            data=copy(data1)
            labels=copy(labels1)
            rows=np.where(labels=='healthy')[0]
            rows2=np.where(labels=='colic')[0]
            print 'number of healthy in this node is ',rows.shape[0]
            print 'number of colic in this node is ',rows2.shape[0]
            if rows.shape[0]==0 or rows2.shape[0]==0:
                self.decisions[nodeNum]=(rows.shape[0],rows2.shape[0])
                return
            
            IGA=[]
            thresholds=[]
            for attr in range(0,16):
                thresh,IG=self.findThresholdAndIG(data,attr,labels)
                IGA.append(IG)
                thresholds.append(thresh)
                
            maxIG=max(IGA)
            Attr=IGA.index(maxIG)
            print 'Attr is ',Attr
            thresh=thresholds[Attr]
            self.usedThresholds[Attr].add(thresh)
            self.Tree[nodeNum]=Attr
            self.Thresholds[nodeNum]=thresh
            rows=np.where(data[:,Attr]>=thresh)[0]
            rows2=np.where(data[:,Attr]<thresh)[0]
            
            dataLeft=copy(data[rows])
            dataRight=copy(data[rows2])
            #if rows.shape[0]==0 or rows2.shape[0]==0:
                #return
            
            labelsLeft=copy(labels[rows])
            labelsRight=copy(labels[rows2])
            print '\n\n'
            self.contructTree(dataLeft,2*nodeNum,labelsLeft)
            self.contructTree(dataRight,2*nodeNum+1,labelsRight)

            
            
            
            
                
        
    
    def loadTrain(self):
        
        f=open('horseTrain.txt')
        
        for line in f:
           
            self.AllValues={}
            
            line=line.rstrip()   
            line=line[0:len(line)-1]
            attrs=line.split(',')
            #print ' before attrs is ',attrs
            attr2=[float(i) for i in attrs[0:len(attrs)-1]]
            #attr2.append(attrs[-1])
            self.labels.append(attrs[-1])
            #print 'attr2 is ',attr2,' and type is ',type(attr2[0])
            
            attrs=copy(np.asarray(attr2))
            attrs=attrs.reshape(1,16)
            #print 'attrs is ',attrs
            #print 'self.train.shape[0] is ',self.train.shape[0]
            if self.train.shape[0]==0:
                self.train=copy(attrs)
            else:
                self.train=copy(np.vstack((self.train,attrs)))
                
        self.labels=copy(np.asarray(self.labels))
        self.labels=copy(self.labels.reshape(-1,1))
        #print 'train is ',self.train
        #print 'labels are ',self.labels
        print 'train set is ',self.train.shape
        print 'labels set is ',self.labels.shape
        print 'Now calling contructTree'
        
        self.contructTree(self.train,1,self.labels)
        print 'the tree is '
        for i in range(1,20): 
            
            print self.Tree[i],' '
        print 'the thresholds are '
        for i in range(1,20): 
            
            print self.Thresholds[i],' '
        print 'self.decisions is ',self.decisions
        
        
        self.test(self.train,self.labels)
    
    def loadTest(self):
        
        f=open('horseTest.txt')
        
        for line in f:
           

            
            line=line.rstrip()   
            line=line[0:len(line)-1]
            attrs=line.split(',')
            #print ' before attrs is ',attrs
            attr2=[float(i) for i in attrs[0:len(attrs)-1]]
            #attr2.append(attrs[-1])
            self.labelsT.append(attrs[-1])
            #print 'attr2 is ',attr2,' and type is ',type(attr2[0])
            
            attrs=copy(np.asarray(attr2))
            attrs=attrs.reshape(1,16)
            #print 'attrs is ',attrs
            #print 'self.train.shape[0] is ',self.train.shape[0]
            if self.Test.shape[0]==0:
                self.Test=copy(attrs)
            else:
                self.Test=copy(np.vstack((self.Test,attrs)))
                
        self.labelsT=copy(np.asarray(self.labelsT))
        self.labelsT=copy(self.labelsT.reshape(-1,1))
        
        print '\n\nnow testing the test set \n'
        self.test(self.Test,self.labelsT)
    
    
    def checkAccuracy(self,gold,predicted):
        gold=copy(gold.tolist())
        predicted=copy(predicted.tolist())
        
        correct=0
        for i in range(0,len(gold)):
            #print 'gold[i]= ',gold[i],' & predicted[i] =',predicted[i]
            if gold[i][0]==predicted[i]:
                correct+=1
        
        return 100*(float(correct)/len(gold))
        
    def findLabel(self,data1,nodeNum):
        data=copy(data1)
        #print 'nodeNum is ',nodeNum
       # print 'self.Tree[nodeNum is',data[self.Tree[nodeNum][0]]
        if self.Tree[nodeNum][0]==-1:
            #then we check the decisions 
            healthy,colic=self.decisions[nodeNum]
            if healthy>0:
                res= 'healthy'
            else:
                res= 'colic'

        elif data[self.Tree[nodeNum][0]]>=self.Thresholds[nodeNum][0]:
            #go left
            res=self.findLabel(data,2*nodeNum)
        else:
            res=self.findLabel(data,2*nodeNum+1)
           
        #print 'returning ',res   
        return res
                    
    def test(self,data1,labels1):
        data=copy(data1)
        labels=copy(labels1)
        predicted=[]
        for i in range(0,data.shape[0]):
            
            res=self.findLabel(data[i],1)
            predicted.append(res)
            print 'testing ',i,' predicted= ',res,' gold is ',labels[i][0]
            
            
        predicted=np.asarray(predicted)
        acc=self.checkAccuracy(labels,predicted)
        print 'Accuracy is ',acc,'%'
                        
       
    #def checkValues(self):
        
ob1=DecisionTree()
ob1.loadTrain()
ob1.loadTest()

