# NHL Salary Predictor v0.1

The purpose of this project is to develop a model to predict NHL Player Salaries.
In a sense, the prediction is really providing a guideline for fair market value
for a player.  The model performs better than Linear Regression which is good but 
I wasn't fully satisifed with the performance, so I dug a little deeper.  It turns 
out, the model is better at predicting Cap Hit than Salary, which I believe to be 
more relevant to an NHL GM anyways.  I hope to continue improving the model and 
tweaking it to see how it adapts to different scenarios.  In the event you were an NHL
GM and wanted to maximize the potential of this model heading into free agency, 
you'd definitely want to rebuild the model on an updated dataset after every season
due to the nature of the (usually) increasing salary cap.  That said, this model
utilizes data from the 2016/2017 season (I've sourced the 2017/2018 data from
hockeyabstract for a later update.)  Due to the pandemic, the 2019/2020 and 
2020/2021 season have been greatly affected in a number of ways (ex. Salary Cap), 
so I'm fine with sticking to 2017/2018 data while tweaking the model for now.

## Files

* data.csv - NHL player data with statistics and salaries from the 2016/2017 season
* SalaryPredictor.py - Where the magic happens. The data is split into train/test 
(random_state set to 0 so it's always the same) and the machine learning model is 
built and then tested against the test split
* OLSModel.py - Ordinary Least Squares Regression Model to predict player 
salaries (for comparison purposes)
* OLSModelNoPCA.py - Ordinary Least Squares Regression model (without Principal 
Component Analysis, so it doesn't account for multicollinearity)
* Presentation.pdf - Accompanying model building and results presentation/report
* README.md

## To Do's

* Clean up column (stat) deletion
* Save model to h5 file (right now SalaryPredictor builds a new model and evaluates each time)
* Write script to run created h5 model on datasets
* Utilize Salary Cap Hit to build model and make predictions on instead of base salary !!!
* Convert output of predictions to Excel file
* Utilize OLS results to drop more predictor variables (P values > 0.05)
* Further tuning of optimizer, learning rate and model (ex. try boosting)
* Add file to display tuning code/process
* Update train/test dataset (have 2017/2018 updated stats from hockeyabstract)
* Compare predictions to free agent signings
* Clean up overall code
* Move presentation information into README (model building, performance, etc.)

## Future Ideas

* Try to convert salary value to scaled number (1-10) to rank player performance
* Apply model to CHL Player's and see what happens (will have to simplify based on available CHL Player stats)

## Built With

* [pandas](https://pandas.pydata.org/) 
* [Matplotlib](https://matplotlib.org/)
* [NumPy](https://numpy.org/) 
* [Keras](https://keras.io/) - Python interface for TensorFlow
* [TensorFlow](https://www.tensorflow.org/)  
* [sklearn](https://scikit-learn.org/stable/) 
* [seaborn](https://seaborn.pydata.org/) 

## Acknowledgments

* Hockeyabstract for the original raw data excel sheet (http://www.hockeyabstract.com/)
* Cam Nugent, for posting the original challenge and dataset on Kaggle (https://www.kaggle.com/camnugent/predict-nhl-player-salaries)
* CapFriendly (https://www.capfriendly.com/) for Salary Cap Hit information found in presentation file
