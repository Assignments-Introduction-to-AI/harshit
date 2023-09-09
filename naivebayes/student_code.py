import math
import re

class Bayes_Classifier:

    def __init__(self):
        self.class_probabilities = {}  # Stores P(class)
        self.word_probabilities = {}   # Stores P(word|class)
        self.vocab = set()            # Vocabulary set
        self.alpha = 1                

    def train(self, lines):
        # Process and tokenize the dataset
        positive_reviews = []
        negative_reviews = []
        for line in lines:
            rating,customer, text = line.split('|')
            if rating == '5':
                positive_reviews.append(text)
            elif rating == '1':
                negative_reviews.append(text)

        # Calculate class probabilities
        total_reviews = len(positive_reviews) + len(negative_reviews)
        self.class_probabilities['positive'] = len(positive_reviews) / total_reviews
        self.class_probabilities['negative'] = len(negative_reviews) / total_reviews

        # Process reviews and update vocabulary
        self.vocab.update(self.tokenize(positive_reviews))
        self.vocab.update(self.tokenize(negative_reviews))

        # Calculate word probabilities using TF-IDF
        for word in self.vocab:
            tf_positive = self.tf(word, positive_reviews)
            tf_negative = self.tf(word, negative_reviews)
            idf = self.idf(word, positive_reviews, negative_reviews)
            
            # Calculate TF-IDF score
            tfidf_positive = tf_positive * idf
            tfidf_negative = tf_negative * idf

            self.word_probabilities[(word, 'positive')] = tfidf_positive
            self.word_probabilities[(word, 'negative')] = tfidf_negative

    def classify(self, lines):
        predictions = []
        for line in lines:
            positive_prob = math.log(self.class_probabilities['positive'])
            negative_prob = math.log(self.class_probabilities['negative'])

            # Tokenize and process the review
            words = self.tokenize([line])
            for word in words:
                if (word, 'positive') in self.word_probabilities:
                    positive_prob += math.log(max(self.word_probabilities[(word, 'positive')], 1e-10))  # Add epsilon to avoid log(0)
                if (word, 'negative') in self.word_probabilities:
                    negative_prob += math.log(max(self.word_probabilities[(word, 'negative')], 1e-10))  # Add epsilon to avoid log(0)

            # Classify based on probabilities
            if positive_prob > negative_prob:
                predictions.append('5')
            else:
                predictions.append('1')

        return predictions


    def tokenize(self, lines):
        # Tokenize, remove punctuation, lowercase, and remove stop words
        stop_words = set(['call', 'upon', 'still', 'nevertheless', 'down', 'every', 'forty', '‘re', 'always', 'whole', 'side', "n't", 'now', 'however', 'an', 'show', 'least', 'give', 'below', 'did', 'sometimes', 'which', "'s", 'nowhere', 'per', 'hereupon', 'yours', 'she', 'moreover', 'eight', 'somewhere', 'within', 'whereby', 'few', 'has', 'so', 'have', 'for', 'noone', 'top', 'were', 'those', 'thence', 'eleven', 'after', 'no', '’ll', 'others', 'ourselves', 'themselves', 'though', 'that', 'nor', 'just', '’s', 'before', 'had', 'toward', 'another', 'should', 'herself', 'and', 'these', 'such', 'elsewhere', 'further', 'next', 'indeed', 'bottom', 'anyone', 'his', 'each', 'then', 'both', 'became', 'third', 'whom', '‘ve', 'mine', 'take', 'many', 'anywhere', 'to', 'well', 'thereafter', 'besides', 'almost', 'front', 'fifteen', 'towards', 'none', 'be', 'herein', 'two', 'using', 'whatever', 'please', 'perhaps', 'full', 'ca', 'we', 'latterly', 'here', 'therefore', 'us', 'how', 'was', 'made', 'the', 'or', 'may', '’re', 'namely', "'ve", 'anyway', 'amongst', 'used', 'ever', 'of', 'there', 'than', 'why', 'really', 'whither', 'in', 'only', 'wherein', 'last', 'under', 'own', 'therein', 'go', 'seems', '‘m', 'wherever', 'either', 'someone', 'up', 'doing', 'on', 'rather', 'ours', 'again', 'same', 'over', '‘s', 'latter', 'during', 'done', "'re", 'put', "'m", 'much', 'neither', 'among', 'seemed', 'into', 'once', 'my', 'otherwise', 'part', 'everywhere', 'never', 'myself', 'must', 'will', 'am', 'can', 'else', 'although', 'as', 'beyond', 'are', 'too', 'becomes', 'does', 'a', 'everyone', 'but', 'some', 'regarding', '‘ll', 'against', 'throughout', 'yourselves', 'him', "'d", 'it', 'himself', 'whether', 'move', '’m', 'hereafter', 're', 'while', 'whoever', 'your', 'first', 'amount', 'twelve', 'serious', 'other', 'any', 'off', 'seeming', 'four', 'itself', 'nothing', 'beforehand', 'make', 'out', 'very', 'already', 'various', 'until', 'hers', 'they', 'not', 'them', 'where', 'would', 'since', 'everything', 'at', 'together', 'yet', 'more', 'six', 'back', 'with', 'thereupon', 'becoming', 'around', 'due', 'keep', 'somehow', 'n‘t', 'across', 'all', 'when', 'i', 'empty', 'nine', 'five', 'get', 'see', 'been', 'name', 'between', 'hence', 'ten', 'several', 'from', 'whereupon', 'through', 'hereby', "'ll", 'alone', 'something', 'formerly', 'without', 'above', 'onto', 'except', 'enough', 'become', 'behind', '’d', 'its', 'most', 'n\’t', 'might', 'whereas', 'anything', 'if', 'her', 'via', 'fifty', 'is', 'thereby', 'twenty', 'often', 'whereafter', 'their', 'also', 'anyhow', 'cannot', 'our', 'could', 'because', 'who', 'beside', 'by', 'whence', 'being', 'meanwhile', 'this', 'afterwards', 'whenever', 'mostly', 'what', 'one', 'nobody', 'seem', 'less', 'do', '\‘d', 'say', 'thus', 'unless', 'along', 'yourself', 'former', 'thru', 'he', 'hundred', 'three', 'sixty', 'me', 'sometime', 'whose', 'you', 'quite', '’ve', 'about', 'even', ])
        words = []
        for line in lines:
            line = line.lower()
            line = re.sub(r'[^\w\s]', '', line)
            tokens = line.split()
            tokens = [token for token in tokens if token not in stop_words]
            words.extend(tokens)
        return words

    def tf(self, word, reviews):
        # Calculate term frequency (TF) for a word in a set of reviews
        count = sum(1 for review in reviews if word in review)
        return count / len(reviews)

    def idf(self, word, positive_reviews, negative_reviews):
        # Calculate inverse document frequency (IDF) for a word
        df_positive = sum(1 for review in positive_reviews if word in review)
        df_negative = sum(1 for review in negative_reviews if word in review)
        N = len(positive_reviews) + len(negative_reviews)
        return math.log(N / (df_positive + df_negative + 1))  # Adding 1 for smoothing
