import nltk
import math
import re
import os
import _pickle as cPickle

nltk.download('averaged_perceptron_tagger')

def loadInvertedIndex():
    """ Loads in trained system representation  and returns it as a InvertedIndex object """
    input_filename = input(
        'Please specify the file containing the trained system representation: ')
    with open(input_filename, 'r') as f:
        return cPickle.load(f)


class Token:
    """ A class that stores the IDF value and list of TF values for a given Token """

    def __init__(self):
        # Initialize data members
        # (key,value) => (category name, TF  of token for given category)
        self.TF_dict = {}
        self.doc_count = 0  # count of docs containing token used to compute the token's IDF
        self.IDF = 0        # token's IDF

    def setIDF(self, N):
        """ Computes and sets a token's IDF """
        self.IDF = math.log(float(N)/self.doc_count)


class InvertedIndex:
    """ A class that stores the inverted index generated when training the Text Classification System """

    def __init__(self, token_params='default'):
        # Initialize data members
        # (key,value) => (token, Token object)
        self.inverted_index = {}
        # (key,value) => (category, number of docs belonging to given category)
        self.category_count = {}
        # total number of docs in training corpus
        self.N = 0
        if token_params == 'default':
            # flag for whether or not tokenization should be case-insensitive
            self.CASE_INSENSITIVE = False
            # flag for whether or not tokenization should include a stop list
            self.STOP_LIST = False
            # flag for whether or not tokenization should include POS tagging
            self.POS = False
        else:
            # flag for whether or not tokenization should be case-insensitive
            self.CASE_INSENSITIVE = int(token_params[0])
            # flag for whether or not tokenization should include a stop list
            self.STOP_LIST = int(token_params[1])
            # flag for whether or not tokenization should include POS tagging
            self.POS = int(token_params[2])

    def _getTokens(self, filename):
        """ A helper function that tokenizes a  file and returns the list of tokens """
        with open(filename, 'r') as f:
            file = f.read()
            if self.CASE_INSENSITIVE:
                file = file.lower()
            if self.STOP_LIST:
                stop_list1 = ["a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and",
                                   "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could", "dear",
                                   "did", "do", "does", "either", "else", "ever", "every", "for", "from", "get", "got", "had",
                                   "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i", "if", "in", "into", "is",
                                   "it", "its", "just", "least", "let", "like", "likely", "may", "me", "might", "most", "must", "my",
                                   "neither", "no", "nor", "not", "of", "off", "often", "on", "only", "or", "other", "our", "own",
                                   "rather", "said", "say", "says", "she", "should", "since", "so", "some", "than", "that", "the",
                                   "their", "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us",
                                   "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will",
                                   "with", "would", "yet", "you", "your"]
                stop_list2 = ["of", "the", "in", "for", "at"]
                pattern = re.compile("\\b("+'|'.join(stop_list2)+")\\W", re.I)
                file = pattern.sub("", file)
            token_list = nltk.word_tokenize(file)
            if self.POS:
                token_list = nltk.pos_tag(token_list)
        return token_list

    ###################################################################################################
    #   Training Stage Definitions                                                                    #
    ###################################################################################################

    def buildInvertedIndex(self):
        """ Processes training files to create inverted index with unnormalized weights """
        # Prompt user for input file
        input_filename = input(
            'Please specify the file containing the list of labeled training documents: ')
        print('Training the Text Categorization System  now...')

        # Loop through docs, updating the inverted index (TF values & doc_counts of Token obj
        # and N  and category_count of InvertedIndex obj)
        dir_path = os.path.dirname(os.path.abspath(input_filename))
        with open(input_filename, 'r') as training_file_list:
            for line in training_file_list:
                file, category = line.split()
                self._updateInvertedIndex(dir_path+'/'+file, category)

        # Set IDFs of each Token obj in the inverted index
            self._setIDFs()

    def normalizeWeights(self):
        """ Normalizes TF vectors of inverted index """
        normalization_constants = {
        }  # dictionary to store the normalization constants of each category

        # Loop through the inverted index, accumulating the sum-squared of TF*IDF weights for each category
        for token in self.inverted_index.keys():
            for category in self.inverted_index[token].TF_dict.keys():
                weight = self.inverted_index[token].TF_dict[category] * \
                    self.inverted_index[token].IDF
                normalization_constants[category] = normalization_constants.get(
                    category, 0) + weight**2

        # Take the square-root of the category sum-squared weights to get the normalization constants
        for category in normalization_constants.keys():
            normalization_constants[category] = math.sqrt(
                normalization_constants[category])

        # Loop through the inverted index, normalizing each TF by the appropriate normalization constant
        for token in self.inverted_index.keys():
            for category in self.inverted_index[token].TF_dict.keys():
                self.inverted_index[token].TF_dict[category] /= normalization_constants[category]

    def saveInvertedIndex(self):
        """ Saves an InvertedIndex object using the pickle module """
        # Prompt user for output filename
        output_filename = input(
            'Please specify the name for the file containing the trained system representation: ')

        # Save trained system representation
        with open(output_filename, 'w') as f:
            cPickle.dump(self, f)

    def _updateInvertedIndex(self, file, category):
        """ A helper function that updates the InvertedIndex obj based on the contents of the given file-label pair """
        # Generate list of tokens for the given document and set of unique tokens
        token_list = self._getTokens(file)
        token_set = set(token_list)

        # Iterate through token_list, incrementing TF of given category (label)
        for token in token_list:
            if token in self.inverted_index:
                self.inverted_index[token].TF_dict[category] = self.inverted_index[token].TF_dict.get(
                    category, 0)+1
            else:
                self.inverted_index[token] = Token()
                self.inverted_index[token].TF_dict[category] = 1

        # Iterate through token_set, incrementing doc_count of given token
        for token in token_set:
            self.inverted_index[token].doc_count += 1

        # Increment N (total number of docs in corpus)
        self.N += 1

        # Update category_count
        self.category_count[category] = self.category_count.get(category, 0)+1

    def _setIDFs(self):
        """ A helper function that sets the IDF of each token in the InvertedIndex object """
        for token in self.inverted_index.keys():
            self.inverted_index[token].setIDF(self.N)

    ###################################################################################################
    #   Testing Stage Definitions                                                                     #
    ###################################################################################################

    def categorizeTexts(self):
        """ Categorizes test files """
        # Prompt user for input filename and output filename
        input_filename = input(
            'Please specify the file containing the list of test documents: ')
        output_filename = input(
            'Please specify the name for the output file containing the labeled test documents: ')
        print('Applying the Text Categorization System to the test documents now...')

        # Loop through docs and categorize them
        dir_path = os.path.dirname(os.path.abspath(input_filename))
        with open(input_filename, 'r') as test_file_list:
            with open(output_filename, 'wb') as outfile:
                for doc in test_file_list:
                    self._categorize(dir_path, doc.strip(), outfile)

    def _categorize(self, dir_path, doc, outfile):
        """ Helper function used  to categorize a single document and write the results to the outfile """
        # Generate list of tokens for the given document
        token_list = self._getTokens(dir_path+'/'+doc)

        # Compute similarity metric for each of the categories
        similarities = {}
        for category in self.category_count.keys():
            similarities[category] = self._sim(token_list, category)

        # Pick the category with highest similarity and write results to output file
        label = max(similarities, key=similarities.get)
        outfile.write(doc+' '+label+'\n')

    def _sim(self, token_list, category):
        """ Helper function that computes the actual similarity metric """
        doc_TFs = {}
        similarity = 0
        # Compute TFs of document tokens
        for token in token_list:
            doc_TFs[token] = doc_TFs.get(token, 0)+1

        # Compute similarity metric
        for token in doc_TFs:
            if token in self.inverted_index and category in self.inverted_index[token].TF_dict:
                category_TF = self.inverted_index[token].TF_dict[category]
                doc_TF = doc_TFs[token]
                IDF = self.inverted_index[token].IDF
                similarity += category_TF*doc_TF*(IDF**2)
        return similarity


####################################################################################################
#  Driver Program                                                                                  #
####################################################################################################
if __name__ == "__main__":
    print('Welcome to the Text Categorization Program:')
    print('-------------------------------------------')
    train = input('Would you like to load in a previously trained system (0) or train the text categorization system with a new training set/new parameters (1): ')
    if train:
        tokenization_parameters = input(
            'Would you like to use the default tokenization parameters (0) or select your own (1): ')
        if tokenization_parameters:
            print("Parameters take the form of <case-insensitive><stop_list><POS>. e.g. 101 => case insensitive=True, stop list=False, and POS=True")
            token_params = input(
                'Please enter your choice of tokenization parameters: ')
        else:
            token_params = 'default'
        inverted_index = InvertedIndex(token_params)
        inverted_index.buildInvertedIndex()
        inverted_index.normalizeWeights()

        save = input(
            'Would you like to save a representation of the trained system for future use (0=>no,1=>yes): ')
        if save:
            inverted_index.saveInvertedIndex()

    test = input('Would you like to categorize a test set (0=>no,1=>yes): ')
    if test:
        if not train:
            inverted_index = loadInvertedIndex()
        inverted_index.categorizeTexts()

    if not (train or test):
        print("This probably isn't the program you are looking for.")

    print("Thank you for using the Text Categorization Program!")
