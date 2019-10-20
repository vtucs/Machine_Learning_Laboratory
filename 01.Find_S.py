"""
1. Implement and demonstrate the FIND-S algorithm for finding the most specific
hypothesis based on a given set of training data samples. Read the training data from a
'.CSV' file.
"""

"""
Find-S Algorithm:
Step 1: Initialize h to the most specific hypothesis in H.
Step 2: For each positive training instance x:
            For each attribute constraint a[i] in h
                If the constraint a[i] is satisfied by x
                    Then do nothing
                Else replace a[i] in h by the next more general constraint that is satisfied by x.
Step 3: Output hypothesis h.
"""

import csv

if __name__ == '__main__':
    with open('ds1.csv') as csvFile: #opening the file to manipulate
        data = [line[:-1] for line in csv.reader(csvFile) if line[-1] == "Y"] #Cosidering only positive examples (examples with "Y")
    print("Positive Examples are: {}".format(data)) #Printing the postitive examples taken in last step
    S = ['o'] * len(data[0])  #Initialising the specific hypothesis
    print("Output after each step:\n{}".format(S)) 
    #looping over the positive examples
    for example in data: 
        i = 0
        for feature in example:
            S[i] = feature if S[i] == 'o' or S[i] == feature else '?' #Changing the specific hypothesis according to the algorithm
            i += 1
        print(S) #printing the intermediate specific hypothesis after each iteration

"""
Output:

Positive Examples are: [['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'], ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'], ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change']]
Output after each step:
['o', 'o', 'o', 'o', 'o', 'o']
['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same']
['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']
['Sunny', 'Warm', '?', 'Strong', '?', '?']
"""
