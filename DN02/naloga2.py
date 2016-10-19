import unidecode
from collections import Counter
from my_clustering import HierarchicalClustering
import math


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

    def identify_language(self, filename):
        """
        Function identifies language from given filename
        """
        text = self.read_file(filename)
        pairs = Counter(self.tuples(text))
        results = {k:self.cosine_distance(pairs,v) for k,v in self.languages.items()}
        sorted_results = sorted(results.items(), key=lambda x: x[1])
        return sorted_results

    def print_identification_results(self, original, r):
        print('-------------------------------')
        print('Original text was in: ', original)
        preds = "".join('('+self.lang_ids[r[i][0]]+', '+str(r[i][1])+') ' for i in range(3))
        print('Predicted languages and distances: ', preds)

    @staticmethod
    def cosine_distance(lang1, lang2):
        inter_key = set(lang1.keys()).intersection(set(lang2.keys()))
        dot_prod = float(sum([lang1[key] * lang2[key] for key in inter_key]))
        len_vec1 = float(math.sqrt(sum([math.pow(lang1[key], 2) for key in lang1.keys()])))
        len_vec2 = float(math.sqrt(sum([math.pow(lang2[key], 2) for key in lang2.keys()])))
        return 1 - (dot_prod / (len_vec1 * len_vec2))

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
    # Prvi del naloge
    ls = LanguageSimilarity(['Slovenian', 'Russian', 'Lithuanian', 'Estonian', 'Macedonian', 'Serbian (Latin)',
                             'Polish', 'Greek', 'Bulgarian', 'Icelandic', 'Filipino', 'Italian', 'Latvian',
                             'Luxembourgish', 'German', 'Norwegian (Nynorsk)', 'Swedish', 'Romanian',
                             'Belorus', 'Slovak', 'Bosnian (Cyrillic script)', 'Japanese', 'Corsican',
                             'Chinese', 'Javanese', 'Spanish', 'Portuguese'], 'translations/')
    hc = HierarchicalClustering(ls.languages, ls.lang_ids, HierarchicalClustering.cosine_distance,
                                HierarchicalClustering.average_linkage)
    hc.compute_clusters()  # create clusters
    hc.create_dendrogram()  # create dendrogram based on clusters
    hc.dendro.create_leaves()  # add leaf nodes to dendrogram (single countries)
    hc.prepare_visualization()

    # Drugi del naloge
    real = ['Japanese', 'Swedish', 'Slovenian', 'Spanish', 'German', 'Icelandic']
    for i in range(0,6):
        lang = ls.identify_language('samples/'+str(i)+'.txt')
        ls.print_identification_results(real[i], lang)

