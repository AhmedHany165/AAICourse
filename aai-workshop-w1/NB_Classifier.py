#############################################################################
# NB_Classifier.py
#
# Implements the Naive Bayes classifier for simple probabilistic inference.
# It assumes the existance of data in CSV format, where the first line contains
# the names of random variables -- the last being the variable to predict.
# This implementation aims to be agnostic of the data (no hardcoded vars/probs)
#
# Version: 1.0, Date: 03 October 2022
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################
# 18/11/2024 : refactored by Bartosz Krawczyk for readability.
#############################################################################

import sys
import math

# example usage
# try running these commands in the terminal:
# "python NB_Classifier.py .\data\play_tennis-train.csv .\data\play_tennis-test.csv"
# "python NB_Classifier.py .\data\lung_cancer-train.csv .\data\lung_cancer-test.csv"


# this class is intended to be called by console commands
class NB_Classifier:
    def __init__(self, training_csv_file, testing_csv_file, log_probabilities=False, default_missing_count=0.000001):
        # initialising variables common to most functions
        self.rand_vars = []
        self.rv_key_values = {}
        self.rv_all_values = []
        self.predictor_variable = None
        self.num_data_instances = 0
        # settings
        self.log_probabilities = log_probabilities
        self.default_missing_count = default_missing_count

        # read in training data
        self.read_data(training_csv_file)

        # train the naive bayes classifier
        probabilities = self.estimate_probabilities()
        test_rv_key_values = self.rv_key_values

        # read in testing data
        self.read_data(testing_csv_file)

        # evaluate model performance
        self.test_learnt_probabilities(test_rv_key_values, probabilities)

    # INPUT : a string with a CSV file's reference path
    # EFFECT: populate rand_vars, rv_key_values, rv_all_values, and num_data_instances
    # OUTPUT: none
    def read_data(self, data_file):
        print("\nREADING data file %s..." % data_file)
        print("---------------------------------------")

        # reinitialise variables to be read in
        # (necessary as data is read more than once)
        self.rand_vars = []
        self.rv_key_values = {}
        self.rv_all_values = []
        self.num_data_instances = 0

        # open provided CSV file
        with open(data_file) as csv_file:
            for line in csv_file:

                # first line of CSV
                line = line.strip()

                # read the first line of CSV, the headings
                if len(self.rand_vars) == 0:
                    # set rand_vars to the list of headings from CSV
                    self.rand_vars = line.split(',')
                    # initialise rv_key_values with CSV headings
                    # INFO: rv_key_values will eventually hold every unique discrete value under each heading
                    #       e.g. 'T': ['hot', 'mild', 'cool'], 'H': ['high', 'normal'], ...
                    for variable in self.rand_vars:
                        self.rv_key_values[variable] = []

                # read in the data
                else:
                    values = line.split(',')
                    self.num_data_instances += 1

                    self.rv_all_values.append(values)
                    self.update_variable_key_values(values)

        # set the last variable provided as the variable to be predicted
        self.predictor_variable = self.rand_vars[len(self.rand_vars)-1]

        # console output
        print("RANDOM VARIABLES=%s" % self.rand_vars)
        print("VARIABLE KEY VALUES=%s" % self.rv_key_values)
        print("VARIABLE VALUES=%s" % self.rv_all_values)
        print("PREDICTOR VARIABLE=%s" % self.predictor_variable)
        print("|data instances|=%d" % self.num_data_instances)

    # INPUT : a line of a CSV file, in list() format
    # EFFECT: adds newly observed discrete values to the random variable dictionary
    # OUTPUT: none
    def update_variable_key_values(self, values):
        # iterate through the random variables
        for i in range(0, len(self.rand_vars)):
            # read the key values already stored
            variable = self.rand_vars[i]
            key_values = self.rv_key_values[variable]
            # read the provided value
            value_in_focus = values[i]

            # if the value is new, add it
            if value_in_focus not in key_values:
                self.rv_key_values[variable].append(value_in_focus)

    # INPUT : none
    # EFFECT: writes some debug text to the console
    # OUTPUT: the calculated probability table for each input
    def estimate_probabilities(self):
        # in short: contains every rand_var|predictor_var pair in the dataset currently loaded into the class
        variable_counts = self.count_variables()
        # predictor variable counts, stored separately
        prior_counts = variable_counts[self.predictor_variable]

        print("\nESTIMATING probabilities...")

        probabilities = {}

        for variable, counts in variable_counts.items():
            prob_distribution = {}
            for key, val in counts.items():
                variables = key.split('|')

                # calculating the base probability of the predictor variable occurring
                # (it will always be binary, hence len()==1)
                if len(variables) == 1:
                    probability = float(val/self.num_data_instances)

                # calculating the probability of every other variable key value occurring
                else:
                    probability = float(val/prior_counts[variables[1]])

                # store the calculated likelihood of occurrence with the current key in probabilities
                if self.log_probabilities is False:
                    # typical probability calculation
                    prob_distribution[key] = probability
                else:
                    # convert probability to log probability
                    prob_distribution[key] = math.log(probability)

            probabilities[variable] = prob_distribution

        # display statistics about variables
        for variable, prob_dist in probabilities.items():
            prob_mass = 0
            for value, prob in prob_dist.items():
                prob_mass += prob
            print("P(%s)=>%s\tSUM=%f" % (variable, prob_dist, prob_mass))

        return probabilities

    # obtains counts every variable interaction, the probabilities are calculated with the counts
    # INPUT : none
    # EFFECT: writes some debug text in the console
    # OUTPUT: the counts of each variable, and its conditional probability pairs
    def count_variables(self):
        print("\nESTIMATING variable counts...")

        variable_counts = {}
        for i in range(0, len(self.rand_vars)):
            variable = self.rand_vars[i]

            # special case: counting predictor variable
            if i == len(self.rand_vars)-1:
                variable_counts[variable] = self.get_counts(None)

            # counting a random variable, including its conditional pairs
            else:
                variable_counts[variable] = self.get_counts(i)

        print("variable_counts="+str(variable_counts))

        return variable_counts

    # obtains a complete count of a variable, obtains from rand_vars
    # INPUT : the index of a variable stored in rand_vars
    # EFFECT: none
    # OUTPUT: counts, a dictionary of the count of each var|result pair, for a given variable
    def get_counts(self, variable_index):
        counts = {}

        # get the last row, this is the target variable
        predictor_index = len(self.rand_vars)-1

        # iterate through each "row" of rv_all_values
        for values in self.rv_all_values:
            if variable_index is None:
                # get the value of the predictor variable
                # e.g. "yes" or "0"
                value = values[predictor_index]
            else:
                # get the value of the random variable, conditional to the predictor
                # e.g. "sunny|yes" or "1|0"
                value = values[variable_index]+"|"+values[predictor_index]

            # update the dictionary counts[]
            try:
                counts[value] += 1
            except Exception:
                counts[value] = 1

        # verify counts by checking missing prior/conditional combinations
        # if a combination is found to be missing, fill in the missing value
        # so the rest of the code doesn't throw an error
        if variable_index is None:
            counts = self.check_missing_prior_counts(counts)
        else:
            counts = self.check_missing_conditional_counts(counts, variable_index)

        return counts

    # INPUT : counts (of a predictor variable) (see function get_counts() first)
    # EFFECT: writes some debug text
    # OUTPUT: counts, with missing cases populated
    def check_missing_prior_counts(self, counts):
        # iterates every possible predictor_variable value
        for var_val in self.rv_key_values[self.predictor_variable]:

            # if found to be empty, set to the default value
            if var_val not in counts:
                print("WARNING: missing count for variable=" % var_val)
                counts[var_val] = self.default_missing_count
                print("Setting default value instead...")

        return counts

    # INPUT : counts (of a conditional variable) (see function get_counts() first)
    # EFFECT: writes some debug text in the console
    # OUTPUT: counts, with missing probability cases populated
    def check_missing_conditional_counts(self, counts, variable_index):
        # iterate every possible random_var|predictor_var combination
        variable = self.rand_vars[variable_index]
        for var_val in self.rv_key_values[variable]:
            for pred_val in self.rv_key_values[self.predictor_variable]:
                pair = var_val+"|"+pred_val

                # if found to be empty, set to the default value
                if pair not in counts:
                    print("WARNING: missing count for variables=%s" % pair)
                    counts[pair] = self.default_missing_count
                    print("Setting default value instead...")

        return counts

    # INPUT : test_rv_key_values (every possible key value in the dataset)
    # INPUT : probabilities (a table of the calculated likelihood of every possible key occurring)
    # EFFECT: writes some debug text in the console
    # OUTPUT: none
    def test_learnt_probabilities(self, test_rv_key_values, probabilities):
        print("\nEVALUATING on test data...")

        # iterate over all rows in the test data
        for instance in self.rv_all_values:
            distribution = {}
            print("Input vector=%s" % instance)

            # iterate over all values in the predictor variable
            for predictor_value in test_rv_key_values[self.predictor_variable]:
                prob_dist = probabilities[self.predictor_variable]
                prob = prob_dist[predictor_value]

                # iterate over all instance values except the predictor var.
                for value_index in range(0, len(instance)-1):
                    variable = self.rand_vars[value_index]

                    # recall the estimated probability of the combination
                    # presented in the test instance
                    value = instance[value_index]
                    prob_dist = probabilities[variable]
                    cond_prob = value+"|"+predictor_value

                    # combine the probabilities of each random_var|predictor_var in test instance
                    if self.log_probabilities is False:
                        prob *= prob_dist[cond_prob]
                    else:
                        prob += prob_dist[cond_prob]

                distribution[predictor_value] = prob

            # final console output
            normalised_dist = self.get_normalised_distribution(distribution)
            print("UNNORMALISED DISTRIBUTION=%s" % distribution)
            print("NORMALISED DISTRIBUTION=%s" % normalised_dist)
            print("---")

    # INPUT : a dictionary of variable values and their associated probabilities
    # EFFECT: none
    # OUTPUT: a normalised probability distribution
    def get_normalised_distribution(self, distribution):
        normalised_dist = {}
        prob_mass = 0

        # sum up the total of all probabilities
        for var_val, prob in distribution.items():
            prob = math.exp(prob) if self.log_probabilities is True else prob
            prob_mass += prob

        # divide each probability by the total, to normalise them
        for var_val, prob in distribution.items():
            prob = math.exp(prob) if self.log_probabilities is True else prob
            normalised_prob = prob/prob_mass
            normalised_dist[var_val] = normalised_prob

        return normalised_dist


# console input code
if len(sys.argv) != 3:
    print("USAGE: NB_Classifier.py [train_file.csv] [test_file.csv]")
    exit(0)
else:
    file_name_train = sys.argv[1]
    file_name_test = sys.argv[2]
    NB_Classifier(file_name_train, file_name_test)
