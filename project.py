
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from math import log
from collections import defaultdict
from nltk.corpus import stopwords
import string

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')


# Training

train_file_name = input("Name of labeled list of training file: ")

train_file = open(train_file_name)
lines = train_file.readlines()
num_train_files = len(lines)

# store the number of documents for each category
# Key: Category
category_document_count = defaultdict(lambda: 0, {})

# store count of documents that contain a specific token
# Key: Token
inverted_index_count = defaultdict(lambda: 0, {})

# store the number of documents in each category containing a token
# Key: (Category, Token)
inverted_cat_count = defaultdict(lambda: 0, {})

# iterate through each article
for line in lines:

    line = line.strip()
    train_article_path = line.split(' ')[0]
    category = line.split(' ')[1]

    category_document_count[category] += 1

    # Tokenize
    train_article = open(train_article_path)
    content = train_article.read()
    train_tokenized = word_tokenize(content)

    # Stemming, removes capitalization
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in train_tokenized]

    inverted_index_check = defaultdict(lambda: False)
    inverted_cat_check = defaultdict(lambda: False)

    for token in stemmed_tokens:
        # May not need this
        if not inverted_index_check[token]:
            inverted_index_count[token] += 1
            inverted_index_check[token] = True
        if not inverted_cat_check[(category, token)]:
            inverted_cat_count[(category, token)] += 1
            inverted_cat_check[(category, token)] = True


tokens = inverted_index_count.keys()

tf_idf = dict()

categories = category_document_count.keys()

# Calculate TF*IDF
tf_idf_cat = dict()
normalization_constant = defaultdict(lambda: 0, {})
for token in tokens:
    for category in categories:
        tf_idf_cat[(category, token)] = inverted_cat_count[(
            category, token)] * log(num_train_files / inverted_index_count[token], 10)
        normalization_constant[category] += tf_idf_cat[(category, token)]

# Normalize TF*IDF
for token in tokens:
    for category in categories:
        tf_idf_cat[(category, token)] /= normalization_constant[category]

# Testing

# labeled list of predictions for output
predictions = []

train_file_name = input("Name of file containing list of testing examples: ")
test_list = open(train_file_name)
test_lines = test_list.readlines()

# stop list
stops = set(stopwords.words('english'))

# iterate through each article
for line in test_lines:

    # open the article
    test_file = open(line.strip())

    # tokenization of content in article
    content = test_file.read()
    test_tokenized = word_tokenize(content)

    # stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in test_tokenized]

    # dictionary of tokens in each document
    # Key: Token
    # Value: Count
    test_token_count = dict()

    # dictionary of likelihoods of document being in each category
    # Key: Category
    test_category_prob = defaultdict(lambda: 0, {})

    for token in stemmed_tokens:
        # Ignore numbers
        if token.isdigit() == True:
            continue
        # Ignore punctuations
        elif token in list(string.punctuation) or token.find("'") != -1:
            continue
        else:
            for category in categories:
                if token not in stops:
                    if token in test_token_count:
                        test_token_count[token] += 1
                    else:
                        test_token_count[token] = 1

    num_tokens = sum(test_token_count.values())

    # Calculate max likelhihood category
    for category in categories:
        for token, count in test_token_count.items():
            if (category, token) in inverted_cat_count:
                test_category_prob[category] += tf_idf_cat[(
                    category, token)] * count * log(num_train_files / inverted_index_count[token], 10)**2

    # Store prediction for this article
    prediction = max(test_category_prob, key=test_category_prob.get)
    str = line.strip() + ' ' + prediction + '\n'
    predictions.append(str)


output = input("Name of output file: ")

output_file = open(output, 'w')
for line in predictions:
    output_file.write(line)

output_file.close()
