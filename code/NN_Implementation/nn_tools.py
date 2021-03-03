# import torch
# import nltk
# from KaggleWord2VecUtility import KaggleWord2VecUtility

# def parse_as_w2v(dataset, word2vec_model):

#     xData = []
#     yData = []

#     tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#     # for review in dataset["review"]:
#     #     xData += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
#     # for sent in dataset["sentiment"]:
#     #     yData += [sent]

#     for review in dataset["review"][0:200]:
#         xData += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
#     for sent in dataset["sentiment"][0:200]:
#         yData += [sent]

#     for word in xData:
#         if word in word2vec_model.wv
#             embedding = word2vec_model.wv[word]

#     xTensor = torch.Tensor([ xData for xData in xData ])
#     yTensor = torch.Tensor([ yData for yData in dataset["sentiment"] ])
#     return xTensor, yTensor
