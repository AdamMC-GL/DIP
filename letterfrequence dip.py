import pandas as pd
from functools import reduce

class language_predicter:
    """A class for a bigram matrix of a piece of text where the language needs to be determined and can make a prediction"""
    def __init__(self, matrices):
        """Initialize by giving trained matrices, a variable containing all valid characters and a empty matrix"""
        self.matrices = matrices
        self.all_letters = list("abcdefghijklmnopqrstuvwxyz ")
        self.__empty_df()

    def __empty_df(self):
        """This function resets the matrix"""
        self.df = pd.DataFrame(0, columns=self.all_letters + ["Total"], index=self.all_letters)

    def __predict_fill(self, first_letter, second_letter):
        """This function fill the matrix of the text to be calculated, takes in 2 characters representing
        the first and second letter of the letter combination, then adds 1 at the right spot in the matrix"""
        self.df.at[first_letter, second_letter] += 1
        return second_letter

    def __clean_data(self, text):
        """This function takes a character and returns it if it is an accepted character, filtering those who are not"""
        if text in self.all_letters: return text
        return ""

    def predict(self, text):
        """This function makes a prediction what language a piece of string is. Takes a string of text as input
        and returns a prediction of what language the text is."""
        text = ''.join(list(map(self.__clean_data, text)))  # Removes special characters
        text = ' '.join(text.split())  # removes consecutive whitespaces, replace with single space

        reduce(self.__predict_fill, text)  # fills the matrix of letter combinations frequency, takes text as input
        self.df["Total"] = self.df.sum()  # adds the column of total frequence of combinations of each letter
        self.df = self.df.loc[:, self.all_letters].div(self.df["Total"], axis=0).fillna(0)  # turns each value in percentages

        # Tallies the scores by comparing the trained matrices with the prediction matrix and looks at how often
        # a value is closer to one or the other answer. This is counted and set as the score.
        eng = self.df[ (abs(self.df - self.matrices[0].df) < abs(self.df - self.matrices[1].df))][self.df > 0].count().sum()
        nl = self.df[( abs(self.df - self.matrices[0].df) > abs(self.df - self.matrices[1].df))][self.df > 0].count().sum()

        self.__empty_df()  # empties the matrix for the next input
        if nl > eng:  # looks at which score is higher and gives a returns based on it
            return "NL"
        return "ENG"

class trainedmatrix:
    """This class for a bigram matrix of a certain language"""
    def __init__(self, name):
        """Initialize by giving a name, a variable containing all valid characters and an empty matrix"""
        self.name = name
        self.all_letters = list("abcdefghijklmnopqrstuvwxyz ")
        self.df = pd.DataFrame(0, columns=self.all_letters + ["Total"], index=self.all_letters)

    def __clean_data(self, text):
        """This function takes a character and returns it if it is an accepted character, filtering those who are not"""
        if text in self.all_letters: return text
        return ""

    def __fill(self, first_letter, second_letter):
        """This function fill the matrix of the text to be trained with, takes in 2 characters representing
        the first and second letter of the letter combination, then adds 1 at the right spot in the matrix"""
        self.df.at[first_letter, second_letter] += 1
        return second_letter

    def train_matrix(self, text):
        """This function takes a piece of text and uses it to train the program. After training
        you have a matrix build out of the frequence of letter combinations."""
        print("Training matrix: {}".format(self.name))
        # clean data
        text = ''.join(list(map(self.__clean_data, text)))  # Removes special characters
        text = ' '.join(text.split())  # removes consecutive whitespaces, replace with single space

        #fill data
        reduce(self.__fill, text)  # fills the matrix of letter combinations frequency, takes text as input
        self.df["Total"] = self.df.sum()  # adds the column of total frequence of combinations of each letter
        self.df = self.df.loc[:, self.all_letters].div(self.df["Total"], axis=0).fillna(0)  # turns each value in percentages



if __name__ == "__main__":
    # make classes for 2 bigram matrices
    ENmatrix = trainedmatrix("English")
    NLmatrix = trainedmatrix("Dutch")

    # train the matrix for English
    with open('alice.txt', 'r') as file:
        text = file.read().replace('\n', ' ').lower()
    ENmatrix.train_matrix(text)

    # train the matrix for Dutch
    with open('verhaal.txt', 'r') as file:
        text = file.read().replace('\n', ' ').lower()
    NLmatrix.train_matrix(text)

    # make predictions on the test phrases
    with open('testzinnen.txt', 'r') as file:
        testdata = file.read().lower().splitlines()

    print("Making predictions...")
    predicter = language_predicter([ENmatrix, NLmatrix])
    result = list(map(predicter.predict, testdata))

    # results
    print("Amount predicted as English: {}".format(result.count("ENG")))
    print("Amount predicted as Dutch: {}".format(result.count("NL")))
