import gensim
from gensim.models import word2vec
import numpy
import numbers as Number
import logging


class learning(dict):
    # Pickleable class to replicate the functionality of collections.defaultdict
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def __missing__(self, key):
    value = self[key] = []
    return value


def __add__(self, x):
    '''Override addition for numeric types when self is empty'''
    if not self and isinstance(x, Number):
        return x
    raise ValueError


def __sub__(self, x):
    '''Also provide subtraction method'''
    if not self and isinstance(x, Number):
        return -1 * x
    raise ValueError


    # sentences = word_2_vec.Text8Corpus('text8')
    # model = word_2_vec.Word2Vec(sentences, size = 200)
    # model = KeyedVectors.load_word2vec_format('text8-model', binary=False)
    # model.wv.save_word2vec_format('text8_model.txt', binary=False)

    # print(model.doesnt_match("breakfast cereal dinner lunch".split()))
    # print(model.most_similar(positive=['mother', 'son'], negative=['father'], topn=2))
    # print(model.similarity('man', 'woman'))
    # print(model.similar_by_word('engineer', topn=5))

    # Defining function to read all words and vectors from our file


def word_vector_matrix(model_file_name, num_words):
    word_array = []
    vector_array = []
    with open(model_file_name, 'r') as model_file:
        for index, row in enumerate(model_file):
            split_row = row.split()
            word_array.append(split_row[0])
            vector_array.append(numpy.array([float(i) for i in split_row[1:]]))

            if index == num_words:
                return numpy.array(vector_array), word_array

    return numpy.array(vector_array), word_array

    # Read in the labels array and clusters label and return the set of words in each cluster


def find_word_clusters(word_array, cluster_labels):
    cluster_to_words = learning()
    for c, i in enumerate(cluster_labels):
        cluster_to_words[i].append(word_array[c])
    return cluster_to_words

    # Applying KMeans model


if __name__ == "__main__":
    model_file_name = "text8_model.txt"
    num_words = 10000
    reduction_factor = 0.05
    clusters_to_make = int(num_words * reduction_factor)
    vector_array, word_array = word_vector_matrix(model_file_name, num_words)
    kmeans_model = KMeans(init='k-means++', n_clusters=clusters_to_make, n_init=10)
    kmeans_model.fit(vector_array)

    cluster_labels = kmeans_model.labels_
    cluster_inertia = kmeans_model.inertia_
    cluster_to_words = find_word_clusters(word_array, cluster_labels)

    for c in cluster_to_words:
        print(cluster_to_words[c])

    for c in cluster_labels:
        print(c)
