import unidecode
from collections import Counter

__author__ = "Anze Medved, 63120191"
__email__ = "am1947@student.uni-lj.si"
__course__ = "Poslovna inteligenca"


class LanguageSimilarity:

    languages = {}
    directory = ""

    def __init__(self, languages, dir):
        self.directory = dir
        self.langs = self.parse_input(languages)

    def parse_input(self, languages):
        """
        Function reads input files and parses them to dictionary
        :param langs: list of languages
        :return: void
        """
        for lang in languages:
            text = LanguageSimilarity.read_file(self.directory+lang+'.txt')
            trip = Counter(self.tuples(text))
            self.languages[lang] = trip


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
        :param filename: name of file to be read
        :return: file content
        """
        cont = open(filename).read()
        cont = unidecode.unidecode(cont)
        cont.lower()
        cont.replace('\n',' ')
        cont.replace('  ',' ')
        return "".join(c for c in cont if c.isalpha() or c == ' ')