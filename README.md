# Machine Learning Foundations

> I'm using this repository for remember details and saving little code
> snippets related with ML.

## Machine Learning Processes

 1.  Data Collection
	    
	DATA: Accuracy, Relevance, Variability, Quantity, Ethics
    
 2. Data Exploration
    
 3. Data Preparation (Preprocessing)
		 
		 Data Cleaning : Handle Missing values (Deletion, Imputation, Create new category such as 'unknown', 'N/A'), Remove Irrelevant Data
		 Data Normalization: Same Range
		 Feature Selection: Create subset ( Reducing dimensionality, reducing overfitting, improve accuracy, improve training time) using optimization algorithms such as heuristic algorithms, genetic, best first search, greedy .... etc. 
		 Data Splitting: Train-Test
		 Data Augmentation
		 
 4. Modelling
 
		Decision Tree: Small number features
		Random Forest: High number features and complex interactions between features
		Naive Bayes:  High number features
		Lineer Regression: Numerical , Continuous Values
		Logistic Regression: Categorical Values , Binary outcome (True/False, Yes/No),
		
 6. Evaluation
		
		True Positives (TP): Number of samples  _correctly_ predicted as “positive.”
		False Positives (FP): Number of samples  _wrongly_ predicted as “positive.”
		True Negatives (TN): Number of samples  _correctly_ predicted as “negative.”
		False Negatives (FN): Number of samples  _wrongly_ predicted as “negative.”

		Precision= TP / ( TP + FP)   * accuracy of positive predictions
		Recall= TP / (TP+FN) * sensitivity or true positive rate
		F1= 2*Precision*Recall / ( Precision + Recall)

 7. Actionable Insights

> Overfitting Problem: K-Fold Cross Validation, Early Stopping, Pruning ( Feature Selection) , Data Augmentation



## Unsupervised Learning: Clustering, Association

A sample case for clustering:	 
	
	**Case:** Customer Segmentation for an E-commerce Company

	**Background:**
	An e-commerce company wants to improve its marketing strategies by targeting different customer segments more effectively. They have collected a dataset containing information about their customers, including demographics, purchase history, and browsing behavior. The company wants to cluster the customers based on their similarities and differences, with the goal of identifying distinct customer segments.

	**Objective:**
	The objective of this case is to perform customer segmentation using clustering techniques to identify homogeneous groups of customers with similar characteristics. The identified segments will help the company tailor their marketing campaigns, personalize product recommendations, and provide a better shopping experience for each customer segment.

	**Data Description:**
	The dataset provided by the e-commerce company contains the following features for each customer:
	Age: The age of the customer.
	Gender: The gender of the customer.
	Annual Income: The annual income of the customer.
	Purchase History: The total amount spent by the customer on the e-commerce platform.
	Browsing Behavior: The average time spent by the customer on the website per session.
	Tasks:

	**Data Exploration:** Perform an initial exploration of the dataset to gain insights into the distribution of the features, identify missing values, and check for outliers.

	**Data Preprocessing:** Preprocess the dataset by handling missing values, outliers, and any necessary feature scaling or normalization.

	**Feature Selection:** Select relevant features from the dataset based on their importance in differentiating customers.

	**Clustering Algorithm Selection:** Choose an appropriate clustering algorithm that suits the characteristics of the dataset and the problem at hand (e.g., K-means, Hierarchical Clustering, DBSCAN).

	**Cluster Analysis:** Apply the selected clustering algorithm to the preprocessed dataset and analyze the resulting clusters. Interpret the characteristics and behaviors of each cluster to understand the customer segments.

	**Evaluation and Validation:** Evaluate the quality of the clustering results using appropriate validation metrics, such as silhouette score or within-cluster sum of squares (WCSS).

	**Visualization:** Visualize the clusters and customer segments using appropriate plots, such as scatter plots, bar charts, or heatmaps, to provide meaningful insights to the stakeholders.

	**Recommendations:** Based on the identified customer segments, provide actionable recommendations to the e-commerce company on how to tailor their marketing strategies, product offerings, and customer engagement for each segment.

	**Deployment:** Prepare a report or presentation summarizing the findings and recommendations, and share it with the stakeholders of the e-commerce company.

	Note: The specific details and complexity of the case may vary based on the available data and the objectives of the e-commerce company.
	

A sample case for association:

	**Case:** Market Basket Analysis for a Retail Store

	**Background:**
	A retail store wants to improve its sales strategies by understanding the relationships between different products purchased by customers. They have collected transactional data that includes the items purchased by customers during their visits to the store. The store wants to perform market basket analysis to identify frequently occurring item combinations and uncover associations between products.

	**Objective:**
	The objective of this case is to perform association analysis on the transactional data to identify product associations and generate actionable insights for the retail store. The identified associations will help the store optimize product placement, develop cross-selling and upselling strategies, and enhance the overall shopping experience.

	**Data Description:**
	The dataset provided by the retail store contains a series of transactions, where each transaction includes a list of items purchased by a customer. The dataset does not include any additional customer information.

	**Tasks:**

	**Data Exploration:** Perform an initial exploration of the dataset to understand the structure and format of the transactional data.

	**Data Preprocessing:** Preprocess the dataset to transform it into a suitable format for association analysis. This involves converting the dataset into a binary matrix representation, where each row represents a transaction, and each column represents an item. The matrix will contain 1 if an item is present in a transaction and 0 otherwise.

	**Association Rule Mining:** Apply an appropriate association rule mining algorithm, such as Apriori or FP-Growth, to discover frequent itemsets and generate association rules. Frequent itemsets are combinations of items that occur together frequently, and association rules describe relationships between itemsets.

	**Rule Generation:** Set appropriate support and confidence thresholds to filter the generated rules. Support indicates the frequency of occurrence of an itemset, and confidence represents the likelihood that an itemset will lead to the purchase of another item.

	**Rule Evaluation:** Evaluate the generated rules using evaluation metrics such as support, confidence, and lift. Support measures the popularity of a rule, confidence measures its reliability, and lift measures the strength of the association between items.

	**Rule Interpretation:** Interpret the generated rules to gain insights into the associations between products. Identify high-confidence and high-lift rules that indicate strong relationships and potential cross-selling opportunities.

	**Visualization:** Visualize the association rules using appropriate techniques such as scatter plots, network graphs, or word clouds to present the relationships between products in an intuitive manner.

	**Recommendations:** Based on the identified associations, provide actionable recommendations to the retail store on how to optimize product placement, design effective cross-selling and upselling strategies, and enhance the overall shopping experience for customers.

	**Deployment:** Prepare a report or presentation summarizing the findings and recommendations, and share it with the stakeholders of the retail store.

	Note: The specific details and complexity of the case may vary based on the available transactional data and the objectives of the retail store.

## Supervised Learning: Classification, Regression

A sample case for classification:

	**Case:** Credit Card Fraud Detection

	**Background:**
	A financial institution wants to enhance its fraud detection system to identify fraudulent credit card transactions accurately. They have a dataset containing historical credit card transactions, including various features such as transaction amount, location, time, and customer information. The institution wants to build a classification model that can effectively classify transactions as either fraudulent or legitimate.

	**Objective:**
	The objective of this case is to develop a classification model that can accurately predict whether a credit card transaction is fraudulent or legitimate. The model will help the financial institution proactively detect and prevent fraudulent activities, minimizing financial losses and protecting their customers.

	**Data Description:**
	The dataset provided by the financial institution contains a list of credit card transactions, including the following features:

	Transaction Amount: The monetary value of the transaction.
	Location: The geographical location where the transaction occurred.
	Time: The timestamp of the transaction.
	Customer Information: Additional information about the customer, such as age, income, and credit history.
	The dataset also includes a target variable indicating whether a transaction is fraudulent (1) or legitimate (0).

	**Tasks:**

	**Data Exploration:** Perform an initial exploration of the dataset to understand the distribution of features, check for missing values, outliers, and class imbalance.

	**Data Preprocessing:** Preprocess the dataset by handling missing values, outliers, and performing feature scaling or normalization as required.

	**Feature Selection:** Select relevant features from the dataset that are informative for fraud detection. This may involve conducting feature importance analysis or using domain knowledge.

	**Model Selection:** Choose an appropriate classification algorithm that suits the characteristics of the dataset and the problem at hand (e.g., Logistic Regression, Decision Trees, Random Forest, Support Vector Machines).

	**Data Split:** Split the dataset into training and testing sets to evaluate the performance of the classification model accurately. Consider using techniques like stratified sampling to handle class imbalance.

	**Model Training:** Train the selected classification model using the training data.

	**Model Evaluation:** Evaluate the performance of the trained model using appropriate evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. Consider using techniques like cross-validation for robust evaluation.

	**Hyperparameter Tuning:** Fine-tune the hyperparameters of the classification model to optimize its performance. Use techniques like grid search or random search to find the optimal hyperparameter values.

	**Model Validation:** Validate the trained and tuned model using the testing data to ensure its generalization ability and assess its performance on unseen data.

	**Interpretation:** Interpret the model results and examine the feature importance to gain insights into the factors contributing to fraud detection.

	**Recommendations:** Provide actionable recommendations to the financial institution based on the classification model's predictions. This may include strengthening fraud prevention measures, enhancing security protocols, or developing real-time monitoring systems.

	**Deployment:** Prepare a report or presentation summarizing the findings, model performance, and recommendations. Share it with the stakeholders of the financial institution.

	Note: The specific details and complexity of the case may vary based on the available credit card transaction data and the objectives of the financial institution.

A sample case for regression:

	**Case:** House Price Prediction

	**Background:**
	A real estate agency wants to develop a model that can accurately predict the prices of houses based on various features. They have collected a dataset containing information about recently sold houses, including factors such as location, size, number of bedrooms, and other relevant attributes. The agency wants to build a regression model that can estimate the sale prices of houses, helping them and their clients make informed decisions.

	**Objective:**
	The objective of this case is to develop a regression model that can accurately predict the sale prices of houses based on their features. The model will help the real estate agency and its clients estimate the market value of houses, negotiate prices, and make informed investment decisions.

	**Data Description:**
	The dataset provided by the real estate agency contains the following features for each house:

	Location: The geographical location of the house.
	Size: The total size of the house in square feet.
	Number of Bedrooms: The number of bedrooms in the house.
	Number of Bathrooms: The number of bathrooms in the house.
	Age: The age of the house in years.
	Neighborhood Amenities: A score indicating the availability of amenities such as parks, schools, and shopping centers near the house.
	Sale Price: The actual sale price of the house.

	**Tasks:**

	**Data Exploration:** Perform an initial exploration of the dataset to understand the distribution of features, check for missing values, outliers, and correlations between variables.

	**Data Preprocessing:** Preprocess the dataset by handling missing values, outliers, and performing feature scaling or normalization as required.

	**Feature Engineering:** Engineer additional features if necessary, such as calculating the price per square foot or creating categorical variables for different neighborhoods.

	**Model Selection:** Choose an appropriate regression algorithm that suits the characteristics of the dataset and the problem at hand (e.g., Linear Regression, Decision Trees, Random Forest, Gradient Boosting).

	**Data Split:** Split the dataset into training and testing sets to evaluate the performance of the regression model accurately. Consider using techniques like stratified sampling if necessary.

	**Model Training:** Train the selected regression model using the training data.

	**Model Evaluation:** Evaluate the performance of the trained model using appropriate evaluation metrics such as mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE), or R-squared.

	**Hyperparameter Tuning:** Fine-tune the hyperparameters of the regression model to optimize its performance. Use techniques like grid search or random search to find the optimal hyperparameter values.

	**Model Validation:** Validate the trained and tuned model using the testing data to ensure its generalization ability and assess its performance on unseen data.

	**Interpretation:** Interpret the model results to gain insights into the factors influencing house prices. Examine the feature importance to understand which features have the most significant impact on the predicted sale prices.

	**Recommendations:** Provide actionable recommendations to the real estate agency based on the regression model's predictions. This may include suggesting optimal pricing strategies, identifying undervalued or overvalued properties, or understanding the drivers of property value in different neighborhoods.

	**Deployment:** Prepare a report or presentation summarizing the findings, model performance, and recommendations. Share it with the stakeholders of the real estate agency.

	Note: The specific details and complexity of the case may vary based on the available house price data and the objectives of the real estate agency.

## Reinforcement Learning: State, Action, Reward

A sample case for reinforcement learning:

	**Case:** Autonomous Vehicle Navigation

	**Background:**
	A company is developing an autonomous vehicle and wants to train it to navigate safely and efficiently in a simulated environment. The vehicle needs to learn how to make decisions such as accelerating, braking, and steering to reach a destination while avoiding obstacles, following traffic rules, and optimizing fuel consumption. The company aims to use reinforcement learning to train the vehicle to improve its navigation skills over time.

	**Objective:**
	The objective of this case is to develop a reinforcement learning model that can train an autonomous vehicle to navigate in a simulated environment. The model should learn to make optimal decisions based on rewards and penalties received during its interactions with the environment, enabling the vehicle to safely and efficiently reach its destination.

	**Environment and Actions:**
	The simulated environment consists of a virtual road network with various elements such as lanes, intersections, traffic lights, and pedestrians. The vehicle can perform a set of actions, including accelerating, decelerating, steering left or right, and maintaining the current speed.

	**Tasks:**

	**Environment Setup:** Set up the simulated environment with appropriate road network elements, traffic conditions, and obstacles.

	**State Representation:** Define a suitable state representation for the vehicle, considering relevant information such as vehicle position, speed, orientation, and distance to surrounding objects.

	**Reward Design:** Design a reward system that encourages the vehicle to reach its destination quickly, follow traffic rules, avoid collisions with obstacles and pedestrians, and optimize fuel consumption. Assign positive rewards for desirable behaviors and negative rewards or penalties for undesirable actions.

	**Model Selection:** Choose an appropriate reinforcement learning algorithm, such as Q-learning, Deep Q-Network (DQN), or Proximal Policy Optimization (PPO), based on the complexity of the environment and the desired level of control.

	**Training Process:** Train the reinforcement learning model using the simulated environment. Allow the vehicle to explore the environment and interact with it, updating the model based on the rewards received. Use techniques like exploration-exploitation trade-off and experience replay to enhance the learning process.

	**Model Evaluation:** Evaluate the performance of the trained model by measuring its ability to navigate the environment safely, follow traffic rules, and reach the destination efficiently. Monitor metrics such as success rate, average time taken, fuel consumption, and collision rate.

	**Fine-tuning and Optimization:** Fine-tune the hyperparameters of the reinforcement learning model to improve its performance and stability. Experiment with different learning rates, discount factors, exploration rates, and neural network architectures (if applicable).

	**Real-world Testing:** Validate the trained model by deploying it in real-world scenarios, either in a controlled environment or through a simulation that closely mimics real-world conditions. Assess the model's ability to generalize and adapt to novel situations.

	**Model Deployment:** Integrate the trained reinforcement learning model into the autonomous vehicle's system, enabling it to make real-time navigation decisions based on the learned policies. Continuously monitor and update the model as new data becomes available or the environment changes.

	**Monitoring and Improvement:** Monitor the performance of the autonomous vehicle in real-world scenarios and gather feedback from users. Use the collected data to identify areas for improvement, update the model accordingly, and iterate on the training process.

	Note: The specific details and complexity of the case may vary based on the desired level of realism, available resources, and the objectives of the company developing the autonomous vehicle.





