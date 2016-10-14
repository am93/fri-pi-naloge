import csv
import math
from itertools import combinations, product
import matplotlib.pyplot as plt
import random
r = lambda: random.randint(0,255)

__author__ = "Anze Medved, 63120191"
__email__ = "am1947@student.uni-lj.si"
__course__ = "Poslovna inteligenca"

class HierarchicalClustering:
    data = None             # raw voting data, key = country name, value = voting vector
    clusters = None         # array of arrays for clusters
    countries = None        # country names, used also for indexing in data
    clustering_trace = []   # trace of clustering procedure (needed for dendrogram)
    dendrogram = None
    mode_cum = True

    def __init__(self, filename, idx_start, idx_end, cumulative=True):
        """
        When object is created, data is read from csv file
        :param filename: name of csv file
        :param idx_start: start index of data
        :param idx_end: last index of data
        """
        file = open(filename, "r", encoding="latin1")
        csv_reader = csv.reader(file)

        # Get voting countries from first row
        first_row = next(csv_reader)
        country_names = [first_row[i].strip() for i in range(idx_start,idx_end)]
        self.check_for_and_sign(country_names)

        # Get voting data for each country
        if cumulative:
            voting_data = {cn:([0]*len(country_names)) for cn in country_names}
            for row in csv_reader:
                idx_country = country_names.index(row[1].strip())
                for i in range(idx_start, idx_end):
                    if row[i] != '':
                        voting_data[country_names[i-idx_start]][idx_country] += int(row[i])
        else:
            voting_data = {cn: [] for cn in country_names}
            for row in csv_reader:
                for i in range(idx_start, idx_end):
                    if row[i] != '':
                        voting_data[country_names[i - idx_start]].append(float(row[i]))
                    else:
                        voting_data[country_names[i - idx_start]].append(None)

        # Assign results to object
        self.data = voting_data
        self.countries = country_names
        self.clusters = [[cn] for cn in country_names]
        self.mode_cum = cumulative

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
        # if we are in cumulative mode (votes summed)
        if self.mode_cum:
            tmp_zip = list(zip(vec1, vec2))
            idx_ignore.sort(key=int, reverse=True)
            for idx in idx_ignore:
                del(tmp_zip[idx])

            return math.sqrt(sum([math.pow(x1 - x2, 2) for (x1, x2) in tmp_zip]))

        # we have complete column of votes
        else:
            diffs = [math.pow((a - b),2) for a, b in zip(vec1, vec2) if a != None and b != None]
            return math.sqrt(sum(diffs) / (len(diffs)+math.pow(1,-10)))

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
        """
        Function creates dendrogram from trace of clustering algorithm
        """
        self.clustering_trace.reverse()
        trace = self.clustering_trace
        first = trace.pop(0)
        self.dendro = Dendrogram(first[0],first[1])

        while len(trace) > 0:
            tmp = trace.pop(0)
            self.dendro.add_child(tmp[0],tmp[1])

    def prepare_visualization(self, num=1, color=False):
        """
        Function visualizes dendrogram using matplotlib
        """
        if not color or num < 2:
            Dendrogram.visualize_dendrogram(self.dendro, '#000000')
        elif color and num > 1:
            Dendrogram.color_dendrogram(self.dendro, num)
        plt.axis([-1, len(self.dendro.country_order), 0, self.dendro.height + 10])
        plt.xticks(range(0, len(self.dendro.country_order)), [c for c in self.dendro.country_order], rotation=90)
        plt.tight_layout()
        plt.show()

    def print_voting_profile(self, country):
        """
        Function prints voting profile of given country
        :param country: name of country (string)
        """
        profile = list(zip(hc.data[country], hc.countries))
        profile.sort(key= lambda x : x[0], reverse=True)
        print(profile)



class Dendrogram:
    country_order = None    # order of countries from left to right
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
        :param parent: cluster predecessor
        :param lchild: previous left sub-cluster
        :param rchild: previous right sub-cluster
        :param x: x coordinate of current cluster
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
        Function adds child to current dendrogram, which is represented as tree structure. To find out, which child we
        should add, we check current value -> if we found match in index zero we go to left, otherwise to right
        :param height: cluster distance
        :param cluster: tuple of clusters which was joined
        :return: void
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
        Add leaf nodes to dendrogram tree structure (single country). This function also sets x coordinates for all
        clusters.
        :return: void
        """
        countries = {c: idx for (idx, c) in enumerate(self.value[0] + self.value[1])}
        self.recursive_leaves(self, countries)
        self.country_order = self.value[0] + self.value[1]

    @staticmethod
    def recursive_leaves(tree,countries):
        """
        Recursive subfunction to add leaves (DFS)
        :param tree: Dendrogram tree
        :param countries: country names ordered from left to right
        :return: void
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
    def visualize_dendrogram(crr, clr):
        """
        Function visualizes dendrogram using matplotlib
        :param crr: current cluster in dendrogram (start with root)
        :param clr: color
        :return: void
        """
        if len(crr.value) == 1:
            return
        else:
            plt.hlines(crr.height, crr.lchild.x_cord, crr.rchild.x_cord, colors=clr)
            plt.vlines(crr.lchild.x_cord, crr.lchild.height, crr.height, colors=clr)
            plt.vlines(crr.rchild.x_cord, crr.rchild.height, crr.height, colors=clr)
            Dendrogram.visualize_dendrogram(crr.lchild, clr)
            Dendrogram.visualize_dendrogram(crr.rchild, clr)

    @staticmethod
    def get_num_clusters(crr, elem_list, num_elem):
        """
        Function returns wanted number of clusters
        :param tree: Dendrogram object
        :param src_list: array of Dendrogram object (aka clusters)
        :param num_elem: number of wanted clusters
        :return: array of Dendrogram objects
        """
        if len(elem_list) < num_elem:
            elem_list.pop(elem_list.index(crr))
            elem_list.append(crr.lchild)
            elem_list.append(crr.rchild)
            new_crr = max(elem_list, key=lambda x: x.height)
            Dendrogram.get_num_clusters(new_crr, elem_list, num_elem)

        return elem_list

    @staticmethod
    def color_dendrogram(dendro, num_clstr):
        """
        Function will draw color dendrogram, where each cluster will have its own color. Number of wanted clusters
        is passed as an argument. Function also draws red line, which represents "cut-off" point.
        :param dendro: root dendrogram object
        :param num_clstr: number of wanted clusters
        :return: void
        """
        first = max([dendro.lchild, dendro.rchild], key=lambda x: x.height)
        clusters = Dendrogram.get_num_clusters(first,[dendro.lchild, dendro.rchild], num_clstr)

        Dendrogram.visualize_dendrogram(dendro, '#000000')
        for c in clusters:
            color = "".join('#%02X%02X%02X' % (r(), r(), r()))
            Dendrogram.visualize_dendrogram(c,color)

        max_height = max(clusters, key=lambda x: x.height).height
        plt.hlines(max_height+1, -1, len(dendro.country_order), 'r')


if __name__ == "__main__":
    hc = HierarchicalClustering('eurovision-final.csv', 16, 63, True) # read data from file
    hc.compute_clusters() # create clusters
    hc.create_dendrogram() # create dendrogram based on clusters
    hc.dendro.create_leaves() # add leaf nodes to dendrogram (single countries)
    hc.prepare_visualization(13, True) # prepare and show visualization
    hc.print_voting_profile('Slovenia')




