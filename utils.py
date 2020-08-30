import spacy
import numpy as np
import os
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torch

nlp = spacy.load('en')


def dream(input, model, iterations, lr):
    """ Updates the image to maximize outputs for n iterations """
    model.eval()
    out, orig_embeddings, hidden = model(input)
    model.train()
    losses = []
    embeddings_steps = []
    embeddings = torch.autograd.Variable(
        orig_embeddings.mean(1), requires_grad=True)
    embeddings_steps.append(embeddings.clone())
    for i in range(iterations):
        out, embeddings, hidden = model.forward_from_embeddings(embeddings)
        loss = hidden.norm()
        embeddings.retain_grad()
        loss.backward(retain_graph=True)
        avg_grad = np.abs(embeddings.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad

        embeddings.data += norm_lr * embeddings.grad.data
        # input.data = clip(input.data)
        model.zero_grad()
        embeddings.grad.data.zero_()
        losses.append(loss.item())
        embeddings_steps.append(embeddings.clone())
    plt.plot(losses)
    plt.title('activation\'s norm vs iteration')
    plt.ylabel('activation norm')
    plt.xlabel('iterations')
    embeddings_steps = torch.cat(embeddings_steps, dim=0).detach().numpy()
    return embeddings_steps


def sentence_to_tensor(sentence, text):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [text.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed)
    tensor = tensor.unsqueeze(1).T
    return tensor


def get_nearnest_words(V, embeddings, k=3, similarity='cosine'):
    sim = None
    if(similarity == 'cosine'):
        embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        V_norm = np.linalg.norm(V, axis=1, keepdims=True)
        # vocab x len
        sim = (embeddings @ V.T) / (embedding_norms  @ V_norm.T + 1e-10)
    elif(similarity == 'euclidean'):
        m, n = embeddings.shape
        embeddings = embeddings.reshape(m, 1, n)
        V = np.expand_dims(V, 0)
        # vocab x len
        sim = np.linalg.norm((embeddings - V), axis=-1)
    idx = sim.argsort(0)
    maxk_words = idx[-k:, :].T
    sim = sim.T
    maxk_sim = []
    for i in range(len(maxk_words)):
        maxk_sim.append(sim[i, maxk_words[i]])

    maxk_sim = np.array(maxk_sim)
    return maxk_words, maxk_sim


def idx_to_words(maxk_words, text):
    similar_words = []
    m, k = maxk_words.shape
    for i in range(m):
        words = []
        for j in range(k):
            word = text.vocab.itos[maxk_words[i][j]]
            words.append(word)
        similar_words.append(words)
    return similar_words


def save_emeddings_to(orig_embeddings, embeddings_steps, path, text):
    embeddings = np.concatenate([orig_embeddings, embeddings_steps], axis=0)
    vocab = text.vocab.itos
    vocab = vocab + \
        list(map(lambda x: f"step_{x+1}", range(len(embeddings_steps))))
    np.savetxt(f"{path}", embeddings, delimiter='\t')
    dir = os.path.dirname(path)
    filename = os.path.join(dir, 'vocab.tsv')
    with open(filename, "w") as f:
        f.write('\n'.join(vocab))


def get_word_embeddings(pos_neg_words, sentences_embedding_steps, text, step=1, debug=False):
    num_sentences, iters, embed_dims = sentences_embedding_steps.shape
    postive_embeddings = []
    for word in pos_neg_words['positive']:
        postive_embeddings.append(
            text.vocab.vectors[text.vocab.stoi[word]].numpy())
    postive_embeddings = np.asarray(postive_embeddings)

    negative_embeddings = []
    for word in pos_neg_words['negative']:
        negative_embeddings.append(
            text.vocab.vectors[text.vocab.stoi[word]].numpy())
    negative_embeddings = np.asarray(negative_embeddings)

    # pdb.set_trace()
    steps = sentences_embedding_steps[step][list(range(0, iters, 5))]

    if(debug):
        pos_words = list(pos_neg_words['positive'])
        neg_words = list(pos_neg_words['negative'])
        vectors = ['word\tlabel'] + list(map(lambda x: x+'\tpos', pos_words)) + list(map(lambda x: x+'\tneg', neg_words)) + list(
            map(lambda x: f"step_{x+1}\tneutral", range(len(steps))))
        with open('./embeddings/vocab.tsv', 'w') as f:
            f.write('\n'.join(vectors))

    return postive_embeddings, negative_embeddings, steps


def visualize_embeddings(pos_neg_words,
                         sentences_embedding_steps,
                         text, step=1,
                         debug=False,
                         tsne_config={'perplexity': 5, 'learning_rate': 10, 'n_iter': 500}):
    pos, neg, steps = get_word_embeddings(pos_neg_words,
                                          sentences_embedding_steps,
                                          text=text,
                                          step=step,
                                          debug=debug)
    colors = ['g']*len(pos) + ['r']*len(neg) + ['#aaaaaa'] * \
        (len(steps)-1) + ['#000000']
    embedding_arr = np.concatenate([pos, neg, steps], axis=0)

    if(debug):
        np.savetxt('./embeddings/embeddings.tsv',
                   embedding_arr,
                   delimiter='\t')
    # reduce dims
    embedding_arr_reduced = TSNE(n_components=2,
                                 **tsne_config).fit_transform(embedding_arr)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    for embedding, color in zip(embedding_arr_reduced, colors):
        x, y = embedding
        ax.scatter(x, y, color=color)
    return embedding_arr_reduced, colors


def make_word_cloud(embeddings_steps, step, prediction, pos_neg_words, orig_embeddings, text, k=10, filename='./output.png'):
    res = np.expand_dims(embeddings_steps[step, :], 0)

    maxk_words, maxk_sim = get_nearnest_words(res, orig_embeddings, k=k)
    similar_words = idx_to_words(maxk_words, text)

    word_dict = {}
    m, k = maxk_words.shape
    for i in range(m):
        for j in range(k):
            word_dict[similar_words[i][j]] = maxk_sim[i, j]
            if(prediction == 'positive'):
                pos_neg_words['positive'].add(similar_words[i][j])
            elif(prediction == 'negative'):
                pos_neg_words['negative'].add(similar_words[i][j])

    wcloud = WordCloud(width=600, height=400,
                       relative_scaling=1.0,
                       background_color='white').generate_from_frequencies(word_dict)

    # Plotting the wordcloud
    wcloud.to_file(filename)
