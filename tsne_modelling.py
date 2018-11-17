from sklearn.manifold import TSNE
import numpy as np


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


filename = 'vectors300.txt'
 
gnglove_vocab = []
gnglove_embed=[]
embedding_dict = {}
 
file = open(filename,'r')
 
for line in file.readlines():
    row = line.strip().split(' ')
    vocab_word = row[0]
    gnglove_vocab.append(vocab_word)
    embed_vector = [float(i) for i in row[1:]] # convert to list of float
    embedding_dict[vocab_word]=embed_vector
    gnglove_embed.append(embed_vector)
  
print('Loaded GLOVE')
file.close()

def tsne_plot():
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in gnglove_vocab: 
        #tokens.append(embedding_dict[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    X = np.array(gnglove_embed[:100])
    #TRANOFORM TO NUMPUY ARRAY
    new_values = tsne_model.fit_transform(X)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
    plt.savefig("tsne_fig.png")

print(type(gnglove_embed))

print("GO TIME")

tsne_plot()

print("done")
