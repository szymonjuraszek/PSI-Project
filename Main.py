import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

with open('data/Adult.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    for line in csv_reader:
        print(line)