# Smart Lead Scoring Engine (Handling Imbalance Dataset)
Analytics VIdhya June 2022

# Problem Statement
A D2C startup develops products using cutting edge technologies like Web 3.0. Over the past few months, the company has started multiple marketing campaigns offline and digital both. As a result, the users have started showing interest in the product on the website. These users with intent to buy product(s) are generally known as leads (Potential Customers).

Leads are captured in 2 ways - Directly and Indirectly.

Direct leads are captured via forms embedded in the website while indirect leads are captured based on certain activity of a user on the platform such as time spent on the website, number of user sessions, etc.

Now, the marketing & sales team wants to identify the leads who are more likely to buy the product so that the sales team can manage their bandwidth efficiently by targeting these potential leads and increase the sales in a shorter span of time.

Now, as a data scientist, your task at hand is to predict the propensity to buy a product based on the user's past activities and user level information.

# Dataset Description
The dataset contain two classes 0 and 1. The number of samples in 0 class is 37163 and in 1 class is 1998 (Imbalance Dataset). For this we have selected performance metric as F1 score as dataset is imbalance.

# Solution approach
1. we tried to balance the dataset by oversampling [SMOTE].
2. Feature Engineering: We replace NaN value of purchase product with 1 when sign-up date is there for client and 0 when sign-up is not there.
3. We changed product purchased [0,1,2,3] to ONE-HOT encoding.
4. Split the training dataset into train and validation [0.8 and 0.2] respectively.
5. z-Normalization is done on all features.
6. Approach 1: Random forest
7. Approach 2: DNN which contain three layers [linear layer + Relu + Linear layer + Relu + sigmoid layer].
    1. To improve DNN results and avoid overfitting applied Dropout and BatchNormalization.
    2. Loss function: Binary cross entropy loss
    3. Optimizer: Adam, lr = 0.0001
8. Performance Metric: F1   

# DNN architecture
![Screenshot](https://github.com/vibhorgupta1993-git/Smart-Lead-Scoring-Engine-Handling-Imbalance-Dataset-/blob/main/Architecture_D2C.png)

