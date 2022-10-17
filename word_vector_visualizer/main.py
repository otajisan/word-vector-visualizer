# -*- coding: utf-8 -*-

import japanize_matplotlib
import numpy as np
from matplotlib import pylab
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


def hello():
    print('hello')
    return 'hello'


def plot(embeddings, labels):
    pylab.figure(figsize=(20, 20))
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom',
        )
    pylab.show()


def run(vector_file_name='vectors.txt', num_points=400):
    print('hello')

    first_line = True
    index_to_word = []
    with open(f'vectors/{vector_file_name}', 'r') as f:
        for row, line in enumerate(f):
            if first_line:
                dim = int(line.strip().split()[1])
                word_vectors = np.zeros((num_points, dim), dtype=float)
                first_line = False
                continue
            line = line.strip()
            word = line.split()[0]
            vec = word_vectors[row - 1]
            for index, vec_val in enumerate(line.split()[1:]):
                vec[index] = float(vec_val)
            index_to_word.append(word)
            if row >= num_points:
                break
    word_vectors = normalize(word_vectors, copy=False, return_norm=False)

    tsne = TSNE(perplexity=40, n_components=2, init='pca', n_iter=10000)
    two_d_embeddings = tsne.fit_transform(word_vectors[:num_points])
    labels = index_to_word[:num_points]

    plot(two_d_embeddings, labels)


if __name__ == '__main__':
    run()
