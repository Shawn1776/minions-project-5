#Predicting Red Hat Business Value

Team name - minions

Team members: Chirag Jain, Sharmin Pathan

Approach: To classify customer potential using a suitable prediction model.

Technologies Used:
-----------------
- Python 2.7
- Apache Spark
- RFE
- Logistic Regression
- Random Forest Classifier

Introduction:
------------
Like most companies, Red Hat is able to gather a great deal of information over time about the behavior of individuals who interact with them. They’re in search of better methods of using this behavioral data to predict which individuals they should approach—and even when and how to approach them.
With an improved prediction model in place, Red Hat will be able to more efficiently prioritize resources to generate more business and better serve their customers.
(This competition was hosted on Kaggle)

Problem Statement:
-----------------
To create a classification algorithm that accurately identifies which customers have the most potential business value for Red Hat based on their characteristics and activities.

Datatset:
--------
The dataset was taken from https://www.kaggle.com/c/predicting-red-hat-business-value/data.
It includes people.csv, act_train.csv and act_test.csv.
The people file contains all of the unique people (and the corresponding characteristics) that have performed activities over time. Each row in the people file represents a unique person. Each person has a unique people_id.

The activity files contain all of the unique activities (and the corresponding activity characteristics) that each person has performed over time. Each row in the activity file represents a unique activity performed by a person on a certain date. Each activity has a unique activity_id.

Preprocessing:
-------------
- Convert the categorical values into numerical ones
- Convert boolean attributes to numerical ones
- Break date column into three separate columns, namely date, month, and year
- Fill '0' for the missing values
- Merge people and activity files
- Separate the data and labels

Flow:
----
- Load the datasets into dataframes
- Perform the preprocessing

Execution:
---------
Ensure the system is up with
- Apache Spark
- RFE
- Matplotlib

The program takes three command line arguments:
1. path to people.csv
2. path to act_train.csv
3. path to act_test.csv

For example: redHat.py people.csv act_train.csv act_test.csv

Performance:
-----------

Tuning the accuracy:
-------------------

Stuff we tried:
--------------
- Plotting graphs for various attributes against the outcome to understand their distribution in the dataset
- PCA Visualization

Challenges:
----------
The dataset was pretty difficult to understand. Enough details were not provided on the competition page about the datasets. 

####References: https://www.kaggle.com/c/predicting-red-hat-business-value
