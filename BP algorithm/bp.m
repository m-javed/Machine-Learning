%% BP main function

% clear

clear all;
%clc;

total_num = 1728;
train_num = 1200;
test_num = total_num - train_num;

% load data
data = csvread('carData.csv');

%random sort between 1 to 1728
k=rand(1,total_num);
[m,n]=sort(k);

% input and output data
input=data(:,1:6);
output1 =data(:,7);

% transform the output from 1 dimension to 4 dimisions
for i=1:total_num
    switch output1(i)
        case 1
            output(i,:)=[1 0 0 0];
        case 2
            output(i,:)=[0 1 0 0];
        case 3
            output(i,:)=[0 0 1 0];
        case 4
            output(i,:)=[0 0 0 1];
    end
end

% ramdomly select train_num samples for training and the remaining test_num samples is for test
trainCharacter=input(n(1:train_num),:);
trainOutput=output(n(1:train_num),:);
testCharacter=input(n(train_num+1:total_num),:);
testOutput=output(n(train_num+1:total_num),:);

% normalize the training input
[trainInput,inputps]=mapminmax(trainCharacter');

%% initialize the parameters

% the initialization of the parameters
inputNum = 6;% the number of nodes in input layer
hiddenNum = 20;% the number of nodes in hidden layer
outputNum = 4;% the number of nodes in output layer

% initialization of weights and bias from input layer to hidden layer and
% those from hidden layer to output layer
w1 = rands(inputNum,hiddenNum);
b1 = rands(hiddenNum,1);
w2 = rands(hiddenNum,outputNum);
b2 = rands(outputNum,1);

% learning rate set to 0.1
yita = 0.1;

%% traing the neural network using bp algorithm
for r = 1:30
    E(r) = 0;% this is the statistical error
    for m = 1:train_num
        % forward propagation
        x = trainInput(:,m);
        % output of the hidden layer
        for j = 1:hiddenNum
            hidden(j,:) = w1(:,j)'*x+b1(j,:);
            hiddenOutput(j,:) = g(hidden(j,:));
        end
        % output of the output layer
        outputOutput = w2'*hiddenOutput+b2;
        
        % calculate the error 
        e = trainOutput(m,:)'-outputOutput;
        E(r) = E(r) + sum(abs(e));
        
        % modify the weights and bias
        
        % this is the modification of weights and bias from hidden layer to
        % output layer
        dw2 = hiddenOutput*e';
        db2 = e;
        
        % this is the modification of weights and bias from input layer to
        % the hidden layer
        for j = 1:hiddenNum
            partOne(j) = hiddenOutput(j)*(1-hiddenOutput(j));
            partTwo(j) = w2(j,:)*e;
        end
        
        for i = 1:inputNum
            for j = 1:hiddenNum
                dw1(i,j) = partOne(j)*x(i,:)*partTwo(j);
                db1(j,:) = partOne(j)*partTwo(j);
            end
        end
        
        w1 = w1 + yita*dw1;
        w2 = w2 + yita*dw2;
        b1 = b1 + yita*db1;
        b2 = b2 + yita*db2;  
    end
end

%% Using the trained neural network to do the classification of the test samples
testInput=mapminmax('apply',testCharacter',inputps);

for m = 1:test_num
    for j = 1:hiddenNum
        hiddenTest(j,:) = w1(:,j)'*testInput(:,m)+b1(j,:);
        hiddenTestOutput(j,:) = g(hiddenTest(j,:));
    end
    outputOfTest(:,m) = w2'*hiddenTestOutput+b2;
end

%% compare the produced result with the correct output
% find out which class the item of the produced result belongs to
for m=1:test_num
    output_fore(m)=find(outputOfTest(:,m)==max(outputOfTest(:,m)));
end

% pridiction error of BP neural network
error=output_fore-output1(n(train_num+1:total_num))';

k=zeros(1,4);  
% find out which class the misclassified item belongs to and calculate the
% misclassified number of each class
for i=1:test_num
    if error(i)~=0
        [b,c]=max(testOutput(i,:));
        switch c
            case 1 
                k(1)=k(1)+1;
            case 2 
                k(2)=k(2)+1;
            case 3 
                k(3)=k(3)+1;
            case 4 
                k(4)=k(4)+1;
        end
    end
end

%calculate the total number of each class in test samples
kk=zeros(1,4);
for i=1:test_num
    [b,c]=max(testOutput(i,:));
    switch c
        case 1
            kk(1)=kk(1)+1;
        case 2
            kk(2)=kk(2)+1;
        case 3
            kk(3)=kk(3)+1;
        case 4
            kk(4)=kk(4)+1;
    end
end

%calculate the right ratio of each class
right_ratio=(kk-k)./kk