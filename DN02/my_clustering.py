import math
from itertools import combinations, product
import matplotlib.pyplot as plt
import random
r = lambda: random.randint(0, 255)

__author__ = "Anze Medved, 63120191"
__email__ = "am1947@student.uni-lj.si"
__course__ = "Poslovna inteligenca"

class HierarchicalClustering:
    data = None             # raw voting data, key = country name, value = voting vector
    clusters = None         # array of arrays for clusters
    keys = {}             # key names, used also for indexing in data
    clustering_trace = []   # trace of clustering procedure (needed for dendrogram)
    distance_fun = None
    linkage_fun = None
    dendrogram = None

    def __init__(self, input_data, keys, distance_fun, linkage_fun):
        self.data = input_data
        self.clusters = [[k] for k in input_data.keys()]
        self.keys = keys
        self.linkage_fun = linkage_fun
        self.distance_fun = distance_fun

    def cosine_distance(self, lang1, lang2):
        inter_key = set(lang1.keys()).intersection(set(lang2.keys()))
        dot_prod = float(sum([lang1[key] * lang2[key] for key in inter_key]))
        len_vec1 = float(math.sqrt(sum([math.pow(lang1[key], 2) for key in lang1.keys()])))
        len_vec2 = float(math.sqrt(sum([math.pow(lang2[key], 2) for key in lang2.keys()])))
        return 1 - (dot_prod / (len_vec1 * len_vec2))

    def average_linkage(self, c1, c2):
        """
        Function computes average linkage between clusters c1 and c2
        For more info see : https://en.wikipedia.org/wiki/UPGMA
        :param c1: first cluster (array of names)
        :param c2: second cluster (array of names)
        :return: average linkage value
        """
        cluster_prod = [(self.data[c1n], self.data[c2n]) for (c1n, c2n) in product(c1,c2)]
        return sum([self.distance_fun(self, *c) for c in cluster_prod]) / (len(c1) * len(c2) * 1.0)

    def complete_linkage(self, c1, c2):
        """
        Function computes complete linkage between clusters c1 and c2 (maximum distance)
        For more info see: https://en.wikipedia.org/wiki/Complete-linkage_clustering
        :param c1: first cluster (array of names)
        :param c2: second cluster (array of names)
        :return: maximum distance
        """
        cluster_prod = [(self.data[c1n], self.data[c2n]) for (c1n, c2n) in product(c1, c2)]
        return max([self.distance_fun(self, *c) for c in cluster_prod])

    def single_linkage(self, c1, c2):
        """
        Function computes single linkage between clusters c1 and c2 (minimum distance)
        For more info see: https://en.wikipedia.org/wiki/Single-linkage_clustering
        :param c1: first cluster (array of names)
        :param c2: second cluster (array of names)
        :return: minimum distance
        """
        cluster_prod = [(self.data[c1n], self.data[c2n]) for (c1n, c2n) in product(c1, c2)]
        return min([self.distance_fun(self, *c) for c in cluster_prod])

    def closest_clusters(self, linkage_fun):
        """
        Function computes closest clusters from current state of clusters. Function takes another function
        as a parameter, which is used to compute linkage. It returns 2 element array, where first value is
        linkage value between clusters and second pair of clusters to be merged.
        :param linkage_fun : function to compute linkage
        :return: [double value, tuple]
        """
        return min([[linkage_fun(self,*comb), comb] for comb in combinations(self.clusters, 2)])

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
            closest = self.closest_clusters(self.linkage_fun)
            self.clustering_trace.append(closest)
            self.update_clusters(closest[1])

    def create_dendrogram(self):
        """
        Function creates dendrogram from trace of clustering algorithm
        """
        self.clustering_trace.reverse()
        trace = self.clustering_trace
        first = trace.pop(0)
        self.dendro = Dendrogram(first[0], first[1])

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
        plt.axis([-1, len(self.dendro.key_order), 0, self.dendro.height + self.dendro.height*0.1])
        plt.xticks(range(0, len(self.dendro.key_order)), [self.keys[c] for c in self.dendro.key_order], rotation=90)
        plt.tight_layout()
        plt.show()




class Dendrogram:
    key_order = None    # order of countries from left to right
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
        self.key_order = self.value[0] + self.value[1]

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
        plt.hlines(max_height+1, -1, len(dendro.key_order), 'r')