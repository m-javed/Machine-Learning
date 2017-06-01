% load data
data = csvread('carData.csv');
count = length(data);

% repeat classification to get the average accurancy of Naive Bayes classifier
repeatTimes = 100;
trainPercent = 0.8;
accurancy = zeros(1, repeatTimes);
for r = 1:repeatTimes
    % divide data into training set and test set
    trainSet = data(1,:);
    testSet = data(2,:);
    for i = 3:count
        if rand(1) < trainPercent
            trainSet(end+1,:) = data(i,:);
        else
            testSet(end+1,:) = data(i,:);
        end
    end
    trainCount = length(trainSet);
    testCount = length(testSet);
    
    % gather the statistics of the training set
    param = zeros(6, 5, 4);
    sum = zeros(1,4);
    for i = 1:trainCount
        t = trainSet(i,:);
        for j = 1:6
            param(j, t(j), t(7)) = param(j, t(j), t(7)) + 1;
        end
        sum(t(7)) = sum(t(7)) + 1;
    end
    for i = 1:4
        param(:, :, i) = param(:, :, i) / sum(i);
    end
    
    % validate the classifier in test set
    testCorrect = 0;
    testFalse = 0;
    for i = 1:testCount
        t = testSet(i,:);
        prob = zeros(1, 4);
        for j = 1:4
            prob(j) = sum(j);
            for k = 1:6
                prob(j) = prob(j) * param(k, t(k), j);
            end
        end
        if prob(t(7))>= prob(1) && prob(t(7))>= prob(2) && prob(t(7))>= prob(3) && prob(t(7))>= prob(4)
            testCorrect = testCorrect + 1;
        else
            testFalse = testFalse + 1;
        end
    end
    accurancy(r) = testCorrect / testCount;
end

% the average accurancy is about 0.8562
result = mean(accurancy(:));