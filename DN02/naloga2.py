import unidecode
from collections import Counter
from my_clustering import HierarchicalClustering


__author__ = "Anze Medved, 63120191"
__email__ = "am1947@student.uni-lj.si"
__course__ = "Poslovna inteligenca"


class LanguageSimilarity:
    lang_ids = {}   # language_id : language name
    languages = {}  # language_id : {pair : value}
    directory = ""  # directory name of input data

    def __init__(self, languages, directory):
        self.directory = directory
        self.get_language_ids(languages)
        self.parse_input()

    def parse_input(self):
        """
        Function reads input files and parses them to dictionary
        """
        for lang_id in self.lang_ids.keys():
            text = LanguageSimilarity.read_file(self.directory+lang_id+'.txt')
            pairs = Counter(self.tuples(text))
            self.languages[lang_id] = pairs

    def get_language_ids(self, languages, index_filename='translations/INDEX.txt'):
        """
        Function get language identifiers from index file.
        :param languages: String names of languages
        :param index_filename: index filename
        """
        ids = {}
        index_file = open(index_filename, 'r')
        for row in index_file:
            splitted = row.split()
            idx = [(splitted[0], lang) for lang in languages if lang in row]
            if len(idx) > 1: print('ERR: Found more than single index !')
            if len(idx) == 1: ids[idx[0][0]] = idx[0][1]
        self.lang_ids = ids

    @staticmethod
    def tuples(string, k=2):
        """
        Function generates tuples from provided string.
        :param string: input string
        :param k: length of tuple (default: 2)
        :return: generator
        """
        for i in range(len(string) - (k - 1)):
            yield string[i:i + k]

    @staticmethod
    def read_file(filename):
        """
        Read file from given filename
        TODO - think if you should keep punctuations
        :param filename: name of file to be read
        :return: file content
        """
        cont = open(filename, encoding="UTF-8").read()
        cont = unidecode.unidecode(cont)
        cont = cont.lower()
        for reg in ['\t', '\n', '  ', '.', ',', ':', ';']:
            cont = cont.replace(reg, ' ')
        return "".join(c for c in cont if c.isalpha() or c == ' ')


if __name__ == "__main__":
    ls = LanguageSimilarity(['Slovenian', 'Russian', 'Albanian', 'Macedonian', 'Serbian (Latin)',
                             'Hungarian', 'Finnish', 'Polish', 'Greek', 'Turkish'], 'translations/')
    hc = HierarchicalClustering(ls.languages, ls.lang_ids, HierarchicalClustering.cosine_distance,
                                HierarchicalClustering.average_linkage)
    hc.compute_clusters()  # create clusters
    hc.create_dendrogram()  # create dendrogram based on clusters
    hc.dendro.create_leaves()  # add leaf nodes to dendrogram (single countries)
    hc.prepare_visualization()
    [print(len(ls.languages[id])) for id in ls.lang_ids.keys()]
