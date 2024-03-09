# Deep Learning Challenge Analysis

The goal of this project is to predict whether applicants will be funded by "Alphabet Soup," a nonexistent nonprofit foundation. This README explains the differences in data preprocessing, model creation, and results between the basic deep learning model and my optimized version. The notebooks were written using Google Colab.

## Data Preprocessing
- The 'IS_SUCCESSFUL' column is the target for this learning model. It has a binary output - either the applicant is funded or they are not.
- In [the first model](deep_learning_challenge.ipynb), 'EIN' and 'NAME' are removed. These columns only contain identifying information, and are therefore unimportant to the model. Any Application Types with 100 or fewer appearances are grouped together in their own category, as well as Classifications with only one appearance. All other columns are left in place to be one-hot-encoded before training.
- Preprocessing for [the optimized model](AlphabetSoupCharity_Optimization.ipynb) follows the same steps for 'APPLICATION_TYPE' and 'CLASSIFICATION.' Values in 'NAME' that have ten or fewer appearances are grouped together in 'Other.' Many organizations have applied hundreds of times, so this helps the model learn which applicants have a history of having their funding approved. I also removed rows with outliers in the 'ASK_AMT' column, though this had little effect on training accuracy.

## The Deep Learning Model
- For both speed and accuracy, two hidden layers seem to work best with this dataset. In the first model, I used 70 neurons in the first layer and 25 in the second. This changes to 50 and 25 in the second model, giving a slight boost to accuracy.
- Given that the output is a simple yes/no, I chose sigmoid as the final activation function. Each of the hidden layers use relu, which proved to be most effective after testing several other functions. Binary Focal Crossentropy was also tested for the loss function. It exchanged a massive drop in loss for a small drop in accuracy, so I decided to stick with Binary Crossentropy.
- Evaluating each model with the testing data shows an 11% decrease in loss and a 6% increase in accuracy when switching to the optimized model. Surprisingly, no extra time is required when testing the optimized model.

I used the plot_keras_history package to quickly plot ten iterations of a ten-epoch training session. The basic model appears on the left, and the optimized model is on the right.


![basic accuracy](https://github.com/owensull12/deep-learning-challenge/assets/143757565/d87df223-7472-42f2-a31e-045606b0205f)![optimized accuracy](https://github.com/owensull12/deep-learning-challenge/assets/143757565/888a57b4-4b03-4a6c-b5fa-b2ae9bee0b20)

![basic loss](https://github.com/owensull12/deep-learning-challenge/assets/143757565/dcef6708-23b2-47cb-808e-460bb189bc9f)![optimized loss](https://github.com/owensull12/deep-learning-challenge/assets/143757565/e3320adb-125f-4c3d-b969-e2d2efa45a8f)
