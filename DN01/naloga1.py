import csv
import math
from itertools import combinations, product
import matplotlib.pyplot as plt

__author__ = "Anze Medved, 63120191"
__email__ = "am1947@student.uni-lj.si"
__course__ = "Poslovna inteligenca"

class HierarchicalClustering:
    data = None             # raw voting data, key = country name, value = voting vector
    clusters = None         # array of arrays for clusters
    countries = None        # country names, used also for indexing in data
    clustering_trace = []   # trace of clustering procedure (needed for dendrogram)
    dendrogram = None

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
        tmp_zip = list(zip(vec1, vec2))
        idx_ignore.sort(key=int, reverse=True)
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
        cluster_prod = [(self.data[c1n],self.data[c2n],[self.countries.index(c1n),self.countries.index(c2n)]) for
                        (c1n, c2n) in product(c1,c2)]
        return sum([self.euclidean_distance(*c) for c in cluster_prod]) / (len(c1) * len(c2) * 1.0)

    def complete_linkage(self, c1, c2):
        """
        Function computes complete linkage between clusters c1 and c2 (maximum distance)
        For more info see: https://en.wikipedia.org/wiki/Complete-linkage_clustering
        :param c1: first cluster (array of names)
        :param c2: second cluster (array of names)
        :return: maximum distance
        """
        cluster_prod = [(self.data[c1n], self.data[c2n], [self.countries.index(c1n), self.countries.index(c2n)]) for
                        (c1n, c2n) in product(c1, c2)]
        return max([self.euclidean_distance(*c) for c in cluster_prod])

    def single_linkage(self, c1, c2):
        """
        Function computes single linkage between clusters c1 and c2 (minimum distance)
        For more info see: https://en.wikipedia.org/wiki/Single-linkage_clustering
        :param c1: first cluster (array of names)
        :param c2: second cluster (array of names)
        :return: minimum distance
        """
        cluster_prod = [(self.data[c1n], self.data[c2n], [self.countries.index(c1n), self.countries.index(c2n)]) for
                        (c1n, c2n) in product(c1, c2)]
        return min([self.euclidean_distance(*c) for c in cluster_prod])

    def closest_clusters(self, linkage_fun):
        """
        Function computes closest clusters from current state of clusters. Function takes another function
        as a parameter, which is used to compute linkage. It returns 2 element array, where first value is
        linkage value between clusters and second pair of clusters to be merged.
        :param linkage_fun : function to compute linkage
        :return: [double value, tuple]
        """
        return min([[linkage_fun(*comb),comb] for comb in combinations(self.clusters, 2)])

    def update_clusters(self, new_cluster):
        """
        Function updates current clusters status based on new cluster merge
        :param new_cluster: new merged cluster (tuple)
        :return: void
        """
        unchanged = [c for c in self.clusters if c not in new_cluster]
        changed = [new_cluster[0] + new_cluster[1]]
        self.clusters = unchanged + changed

    def compute_clusters(self):
        """
        Function computes hierarchical clustering on given data. It stops when number of clusters
        equals 1.
        :return: void - all changes are made to object attributes
        """
        while len(self.clusters) > 1:
            closest = self.closest_clusters(self.average_linkage)
            self.clustering_trace.append(closest)
            self.update_clusters(closest[1])


    def create_dendrogram(self):
        self.clustering_trace.reverse()
        trace = self.clustering_trace
        first = trace.pop(0)
        self.dendro = Dendrogram(first[0],first[1])

        while len(trace) > 0:
            tmp = trace.pop(0)
            self.dendro.add_child(tmp[0],tmp[1])



class Visualizer:
    cluster_trace = []
    countries = {}

    def __init__(self, cluster_trace):
        """
        Initializes new object. Cluster_trace is an array of arrays, where each subarray has to elements -> float value
        that represents distance and tuple of clusters that were merged
        :param cluster_trace: (see above)
        """
        self.cluster_trace = cluster_trace
        self.cluster_trace.reverse()
        tmp_countries = self.cluster_trace[0][1][0] + self.cluster_trace[0][1][1]
        self.countries = {c:idx for (idx, c) in enumerate(tmp_countries)}
        self.cluster_trace.reverse()


    def create_dendrogram(self):
        while len(self.cluster_trace) > 1:
            crr_pair = self.cluster_trace.pop(0)



class Dendrogram:
    country_order = None
    parent = None
    lchild = None
    rchild = None
    height = -1
    x_cord = -1
    value = []

    def __init__(self,height=-1, value=[],  parent=None, lchild=None, rchild=None, x=-1):
        """
        Function initializes new dendrogram object. Object are built in tree like structure, which simplifies visualization.
        :param height: distance between clusters
        :param value: cluster tuple
        :param lchild: previous left sub-cluster
        :param rchild: previous right sub-cluster
        """
        self.lchild = lchild
        self.rchild = rchild
        self.value = value
        self.height = height
        self.parent = parent
        self.x_cord = x

    @staticmethod
    def contains_sublist(lst, sublst):
        """
        Function checks if there exists sublist "sublst" in provided list "lst". It stops searching after first occurance
        of sublist or fails after trying all options.
        :param lst: list in which we search
        :param sublst: list that we search for ("template")
        :return: boolean
        """
        n = len(sublst)
        return any((sublst == lst[i:i + n]) for i in range(len(lst) - n + 1))

    def add_child(self, height, cluster):
        """
        TODO docstring
        :param height:
        :param cluster:
        :return:
        """
        if self.contains_sublist(self.value[0], cluster[0]):
            if(self.lchild is not None):
                self.lchild.add_child(height, cluster)
            else:
                self.lchild = Dendrogram(height,cluster, self)

        elif self.contains_sublist(self.value[1], cluster[0]):
            if(self.rchild is not None):
                self.rchild.add_child(height, cluster)
            else:
                self.rchild = Dendrogram(height,cluster, self)

    def create_leaves(self):
        """
        TODO
        :return:
        """
        countries = {c: idx for (idx, c) in enumerate(self.value[0] + self.value[1])}
        self.recursive_leaves(self, countries)
        self.country_order = self.value[0] + self.value[1]

    @staticmethod
    def print_dendro(dendro):
        if dendro.lchild is not None:
            print(dendro.value)
            print(dendro.x_cord)
            Dendrogram.print_dendro(dendro.lchild)
        else:
            print(dendro.value)
            print(dendro.x_cord)

    @staticmethod
    def recursive_leaves(tree,countries):
        """
        TODO
        :param tree: 
        :param countries: 
        :return: 
        """
        if tree.lchild is not None:
            Dendrogram.recursive_leaves(tree.lchild, countries)
        else:
            tree.lchild = Dendrogram(0, tree.value[0], tree, None, None, countries[tree.value[0][0]])

        if tree.rchild is not None:
            Dendrogram.recursive_leaves(tree.rchild, countries)
        else:
            tree.rchild = Dendrogram(0, tree.value[1], tree, None, None, countries[tree.value[1][0]])

        tree.x_cord = float(tree.lchild.x_cord + tree.rchild.x_cord) / 2


    @staticmethod
    def visualize_dendrogram(crr):
        """
        TODO
        :param crr:
        :return:
        """
        if len(crr.value) == 1:
            return
        else:
            plt.hlines(crr.height, crr.lchild.x_cord, crr.rchild.x_cord)
            plt.vlines(crr.lchild.x_cord, crr.lchild.height, crr.height)
            plt.vlines(crr.rchild.x_cord, crr.rchild.height, crr.height)
            Dendrogram.visualize_dendrogram(crr.lchild)
            Dendrogram.visualize_dendrogram(crr.rchild)




if __name__ == "__main__":
    hc = HierarchicalClustering('eurovision-final.csv', 16, 63)
    hc.compute_clusters()
    hc.create_dendrogram()
    hc.dendro.create_leaves()
    #Dendrogram.print_dendro(hc.dendro)
    Dendrogram.visualize_dendrogram(hc.dendro)
    plt.axis([-1, len(hc.dendro.country_order), 0, hc.dendro.height + 10])
    plt.xticks(range(0,len(hc.dendro.country_order)), [c for c in hc.dendro.country_order], rotation=90)
    plt.tight_layout()
    plt.show()


