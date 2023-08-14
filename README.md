# Fraud-Detection-using-XGBoost
![fraud+prevention](https://github.com/Sourik07/Fraud-Detection-using-XGBoost/assets/113095592/0c50bff8-3399-4293-9984-0174c5f91b19)

In today's digital landscape, fraudulent activities have become a pervasive problem across various domains, including finance, e-commerce, and healthcare. Detecting and preventing fraud is of paramount importance to safeguard the interests of individuals, organizations, and society as a whole. However, the detection of fraud is a challenging task, mainly for the imbalanced nature of the data which is available, where the number of fraudulent instances is significantly lower than non-fraudulent instances. This experiment explores the challenge of detecting fraud in imbalanced datasets. To address this issue, we use Adaptive Synthetic Sampling (ADASYN). ADASYN is utilized to generate synthetic samples for the minority class, augmenting the fraudulent data and balancing the dataset. Once the dataset is properly balanced, we employed XGBoost, a powerful ensemble algorithm, to train the model. Experimental evaluations demonstrate that the combined approach improves fraud detection performance compared to a baseline model trained on the imbalanced data. This research contributes to enhancing fraud detection systems in various domains by effectively handling class imbalance and capturing intricate fraud patterns. We also applied the same oversampling technique on Logistic Regression for a comparision purpose.

What is ADASYN?
One effective oversampling technique is Adaptive Synthetic Sampling (ADASYN), which intelligently creates synthetic samples based on the minority class's density distribution.  One of the challenges in dealing with imbalanced datasets is that traditional oversampling techniques, such as randomly replicating instances from the minority class, can lead to overfitting and poor generalization. ADASYN overcomes this limitation by adaptively generating synthetic samples based on the density distribution of the minority class. It focuses on regions in the feature space where fraudulent instances are underrepresented, ensuring that synthetic samples are generated in areas that require more representation. ADASYN identifies regions in the feature space where fraudulent instances are underrepresented and generate synthetic samples proportionally to the density of the minority class in those regions. This adaptive approach ensures that the synthetic samples focus on areas where fraud instances are scarce, effectively capturing the underlying fraud patterns and improving the model's ability to detect fraudulent instances accurately.

What is XGBOOST?

Gradient-boosting algorithms have become increasingly popular in fraud detection because of their capacity to manage intricate data patterns and deliver precise forecasts. Among these algorithms, XGBoost has emerged as a popular choice. XGBoost, which stands for "Extreme Gradient Boosting," is a machine learning algorithm that is widely used for supervised learning tasks. It belongs to the class of boosting algorithms, which are ensemble learning methods that combine multiple weak predictive models (usually decision trees) to create a strong predictive model.
XGBoost is known for its speed, scalability, and performance in handling structured/tabular data. It can be used for both classification and regression problems. The algorithm works by iteratively training weak learners (decision trees) in a sequential manner, with each subsequent learner trying to correct the mistakes made by the previous ones. The predictions of all the weak learners are then combined to make the final prediction.
The Objective Function of the XGBoost model can be defined as: 
Obj(Θ) = Σ(L(y_i, F(x_i)) + Ω(F) + Φ(Θ))
where:
•	L(y_i, F(x_i)): Loss function measuring the discrepancy between true labels (y_i) and predicted values (F(x_i))
•	Ω(F): Regularization term controlling the complexity of the model (e.g., L1 or L2 regularization)
•	Φ(Θ): Regularization term for the model complexity (e.g., tree structure complexity)

![image](https://github.com/Sourik07/Fraud-Detection-using-XGBoost/assets/113095592/b64735e5-aa86-4f42-811b-e1efec146d21)

Result of our Experiment:

In our experiment, while accuracy provides a general measure of overall correctness, the evaluation of Recall, Precision, and F1 score helps us gain deeper insights into the model's ability to identify fraudulent transactions accurately and minimize false positives. 

![Picture1](https://github.com/Sourik07/Fraud-Detection-using-XGBoost/assets/113095592/a2c882fe-d147-40eb-bda2-e626875a8b76)
![Picture2](https://github.com/Sourik07/Fraud-Detection-using-XGBoost/assets/113095592/959ebc4d-6985-4769-a897-506c3eaea646)


After performing all the experiments we can conclude that the XGBoost model with the Adaptive Synthetic Sampling technique gives the best result. However we can see high accuracy scores in all the models but it is important to note that accuracy might not be the best evaluation metric to assess the performance of your fraud detection model, especially when dealing with imbalanced data. Accuracy can be misleading because even if the model predicts all instances as non-fraudulent, it may achieve a high accuracy due to the majority class's dominance.
The findings revealed that the model trained with XGBoost and ADASYN outperformed the other models in terms of Recall, Precision, and F1 score. This indicates that the XGBoost algorithm, when combined with the ADASYN technique, excels in accurately identifying fraudulent transactions. The higher Recall value indicates a better ability to detect actual fraud cases, while the higher Precision and F1 score indicates a reduced number of false positives and overall improved accuracy. Overall, this research demonstrates the significance of considering both the choice of algorithm and the handling of class imbalance in developing fraud detection models. The combination of XGBoost and ADASYN proves to be a powerful approach, yielding superior performance in accurately identifying fraudulent transactions.

Dataset Link - https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset 
