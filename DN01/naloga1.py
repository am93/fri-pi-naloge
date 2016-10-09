import csv
import math
from itertools import product

__author__ = "Anze Medved, 63120191"
__email__ = "am1947@student.uni-lj.si"
__course__ = "Poslovna inteligenca"

class HierarchicalClustering:
    data = None         # raw voting data, key = country name, value = voting vector
    clusters = None     # array of arrays for clusters
    countries = None    # country names, used also for indexing in data

    def __init__(self, filename, idx_start, idx_end):
        """
        When object is created, data is read from csv file
        :param filename: name of csv file
        :param idx_start: start index of data
        :param idx_end: last index of data
        """
        file = open("eurovision-final.csv", "r", encoding="latin1")
        csv_reader = csv.reader(file)

        # Get voting countries from first row
        first_row = next(csv_reader)
        country_names = [first_row[i].strip() for i in range(idx_start,idx_end)]
        self.check_for_and_sign(country_names)

        # Get voting data for each country
        voting_data = {cn:([0]*len(country_names)) for cn in country_names}
        for row in csv_reader:
            idx_country = country_names.index(row[1].strip())
            for i in range(idx_start, idx_end):
                if row[i] != '':
                    voting_data[country_names[i-idx_start]][idx_country] += int(row[i])

        # Assign results to object
        self.data = voting_data
        self.countries = country_names
        self.clusters = [[cn] for cn in country_names]

    @staticmethod
    def check_for_and_sign(names):
        """
        Function checks array of names and replaces '&' sign with 'and
        :param names: array of strings
        :return: array of strings
        """
        for i in range(0, len(names)):
            if '&' in names[i]:
                names[i] = names[i].replace('&','and')


    def euclidean_distance(self, vec1, vec2, idx_ignore=[]):
        """
        Function computes euclidean distance between given vectors, and ignores
        indices that are in idx_ignore.
        :param vec1 : first vector
        :param vec2 : second vector
        :param idx_ignore: list of indices to ignore
        :return: euclidean distance value (float)
        """
        tmp_zip = zip(vec1, vec2)
        for idx in idx_ignore:
            del(tmp_zip[idx])

        return math.sqrt(sum([math.pow(x1 - x2, 2) for (x1, x2) in tmp_zip]))

    def average_linkage(self, c1, c2):
        """
        Function computes average linkage between clusters c1 and c2
        For more info see : https://en.wikipedia.org/wiki/UPGMA
        :param c1: first cluster (array of names)
        :param c2: second cluster (array of names)
        :return: average linkage value
        """
        combinations = [(self.data[c1n],self.data[c2n],[self.countries.index(c1n),self.countries.index(c2n)]) for
                        (c1n, c2n) in product(c1,c2)]
        return sum([self.euclidean_distance(*c) for c in combinations]) / (len(c1) * len(c2) * 1.0)

    def complete_linkage(self, c1, c2):
        """
        Function computes complete linkage between clusters c1 and c2 (maximum distance)
        For more info see: https://en.wikipedia.org/wiki/Complete-linkage_clustering
        :param c1: first cluster (array of names)
        :param c2: second cluster (array of names)
        :return: maximum distance
        """
        combinations = [(self.data[c1n], self.data[c2n], [self.countries.index(c1n), self.countries.index(c2n)]) for
                        (c1n, c2n) in product(c1, c2)]
        return max([self.euclidean_distance(*c) for c in combinations])

    def single_linkage(self, c1, c2):
        """
        Function computes single linkage between clusters c1 and c2 (minimum distance)
        For more info see: https://en.wikipedia.org/wiki/Single-linkage_clustering
        :param c1: first cluster (array of names)
        :param c2: second cluster (array of names)
        :return: minimum distance
        """
        combinations = [(self.data[c1n], self.data[c2n], [self.countries.index(c1n), self.countries.index(c2n)]) for
                        (c1n, c2n) in product(c1, c2)]
        return min([self.euclidean_distance(*c) for c in combinations])

    def compute_clusters(self):
        """
        Function computes hierarchical clustering on given data. It stops when number of clusters
        equals 1.
        :return: void - all changes are made to object attributes
        """
        while len(self.clusters) > 1:
            a=1 # TODO



if __name__ == "__main__":
    hc = HierarchicalClustering('eurovision-final.csv', 16, 63)

