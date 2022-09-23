# NaiveBayes
Gaussian Naïve Bayes to classify the Spambase data from the UCI ML repository, which can be found here: https://archive.ics.uci.edu/ml/datasets/spambase

-	In the experiment I shuffled the data and divided it into 2 sets of 2300 i.e., 1 train data set and 1 test data set, which contains about 40% spam and 60% not spam in each data set.
-	I calculated the probability of each data set for each class and the probability for spam was about 0.4.
-	Next, I separated the spam and not spam data to calculate mean and standard deviation of train data set.
-	Further, I calculated the mean and standard deviation of each class for train data set and if the standard deviation was 0 then replaced it with 0.0001.
-	Once the mean and standard deviation was calculated, I ran the gaussian naïve bayes on the test data set for both classes spam and not spam.
-	As suggested in the experiment, I took the sum of log of the probabilities for both the classes.
-	Compared the probabilities and if the probability for spam was greater than not spam then marked it as spam else not spam.
-	Once, the predictions for all the data sets were ready, calculated the accuracy, precision, recall and confusion matrix for which the result is as below,

	**Accuracy:  73.78260869565217**
      
    **Precision:  60.87570621468926**
      
     **Recall:  94.62129527991219**
      
     **Confusion matrix:
                      [[835 554]
                       [ 49 862]**
       
-	The accuracy here is above 70%, precision is about 60% and recall about 94%. As we can see in the confusion matrix, false positives are high, so precision is slightly low.
-	The attributes are not independent. Naïve bayes work well here but the accuracy is average, and precision is low.
-	Accuracy and precision can be increased by training some more data sets.


######	Execution Instruction:

**Import libraries:**
  1. Math
  2. Numpy
  3. Pandas
  4. Sklearn.Metrics
  
**Keep the spambase.data file in the same folder as .py file.**

