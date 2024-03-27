# cse5525_humor_nlp

Goal: Can we create a model that can distinguish between a non-humorous, factual statement, and a joke?

For 3/27/24 checkpoint:

running_data.py - prints out dataset containing mix of jokes and non-jokes, with appropriate labels for whether it is a joke or not ("Questions_and_Answers.json")
(This dataset will likely change to include jokes that fit the Q&A format more closely)

TF-IDF_for_Q&A.py - Applied TF-IDF and logistic regression to the Q&A json file, shows super high accuracy but does not really work on real examples
(Could be the problem of the data set or the TF-IDF method or both)

Example outputs provided as PDFs for both running_data.py and TF-IDF_for_Q&A.py. 