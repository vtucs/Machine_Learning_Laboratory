"""
3. Write a program to demonstrate the working of the decision tree based ID3
algorithm. Use an appropriate data set for building the decision tree and apply this
knowledge to classify a new sample.
"""

import math
import csv
from collections import Counter

with open('ds2.csv') as csvFile:
    g_dataset = [tuple(line) for line in csv.reader(csvFile)]
    g_headers = g_dataset[0]
    g_dataset = g_dataset[1:]


class Node:
    def __init__(self, headers, data, attribute=None):
        self.decision_attribute = attribute
        self.child = {}
        self.headers = headers
        self.data = data
        self.decision = None


def get_attribute_column(headers, dataset, attribute):
    i = headers.index(attribute)
    a_list = [ele[i] for ele in dataset]
    return a_list


def calculate_entropy(probs):
    return sum([-prob * math.log(prob, 2) for prob in probs])


def split_data(headers, a_list, attribute, class_value):
    i = headers.index(attribute)
    return [ele for ele in a_list if ele[i] == class_value]


def entropy(headers, a_list, attribute='PlayTennis', gain=False):
    cnt = Counter(get_attribute_column(headers, a_list, attribute))  # Counter calculates the proportion of class
    num_instances = len(get_attribute_column(headers, a_list, attribute))
    probs = [x / num_instances for x in cnt.values()]  # x means count of each attribute.
    if not gain:
        return calculate_entropy(probs)
    gain = 0
    for Class, prob in zip(cnt.keys(), probs):
        gain += -prob * entropy(headers, split_data(headers, a_list, attribute, Class))
    return gain


def information_gain(headers, data):
    max_gain = -1
    max_gain_attribute = None
    for attribute in headers:  # Find max information gain
        if attribute == 'PlayTennis':
            continue
        gain = entropy(headers, data) + entropy(headers, data, attribute, gain=True)
        if gain > max_gain:
            max_gain = gain
            max_gain_attribute = attribute
    return max_gain_attribute


def drop_attribute(headers, dataset, attribute):
    i = headers.index(attribute)
    new_headers = [ele for ele in headers if ele != attribute]
    new_dataset = [tuple(data[:i] + data[i + 1:]) for data in dataset]
    return new_headers, new_dataset


def most_common_outcome(headers, dataset):
    cnt = Counter(get_attribute_column(root.headers, root.data, 'PlayTennis'))
    return cnt.most_common(1)[0][0]


def id3(root):
    if len(root.headers) == 1:
        root.decision = most_common_outcome(root.headers, root.data)
        return

    outcome_value_set = set(get_attribute_column(root.headers, root.data, 'PlayTennis'))
    if len(outcome_value_set) == 1:
        root.decision = list(outcome_value_set)[0]
        return

    max_gain_attribute = information_gain(root.headers, root.data)
    root.decision_attribute = max_gain_attribute
    for attribute in set(get_attribute_column(root.headers, root.data, max_gain_attribute)):
        child_data = split_data(root.headers, root.data, max_gain_attribute, attribute)

        if child_data is None or len(child_data) == 0:
            root.decision = most_common_outcome(root.headers, root.data)
            return

        (new_headers, new_dataset) = drop_attribute(root.headers, child_data, max_gain_attribute)
        root.child[attribute] = Node(new_headers, new_dataset)
        id3(root.child[attribute])


root = Node(g_headers, data=g_dataset)
id3(root)


def print_tree(root, disp=""):
    if root.decision is not None:
        if len(disp) == 0:
            print(str(root.decision))
        else:
            print(disp[:-4] + "THEN " + str(root.decision))
        return
    for attribute, node in root.child.items():
        print_tree(node, disp + "IF {} EQUALS {} AND ".format(root.decision_attribute, attribute))


print("Decision Tree Rules:")
print_tree(root)

"""
Output:

Decision Tree Rules:
IF Outlook EQUALS Rainy AND IF Windy EQUALS True THEN No
IF Outlook EQUALS Rainy AND IF Windy EQUALS False THEN Yes
IF Outlook EQUALS Overcast THEN Yes
IF Outlook EQUALS Sunny AND IF Humidity EQUALS High THEN No
IF Outlook EQUALS Sunny AND IF Humidity EQUALS Normal THEN Yes
"""
