import math
import re

class Bayes_Classifier:
    def __init__(self):
        self.categories = []  
        self.category_counts = {}  
        self.feature_tallies = {}  
        self.words = set()  

    def preprocess_text(self, content):  
        # Remove URLs
        content = re.sub(r'http\S+|www\S+', '', content)
        
        # Remove HTML tags
        content = re.sub(r'<.*?>', '', content)
        
        # Remove non-alphabetic characters and convert to lowercase
        content = re.sub(r'[^a-zA-Z]', ' ', content).lower()
        
        # Tokenize the content
        words = re.findall(r'\b\w+\b', content)
        
        # Remove stop words
        stop_words = set(['the', 'a', 'an', 'is', 'are', 'to', 'in', 'for', 'of'])
        words = [word for word in words if word not in stop_words]

        return words

    def train(self, data):  
        for line in data:
            label, _, content = line.split('|')  
            words = self.preprocess_text(content)
            if label not in self.categories:
                self.categories.append(label)
                self.category_counts[label] = 0
                self.feature_tallies[label] = {}
            self.category_counts[label] += 1
            for word in words:
                self.words.add(word)
                if word not in self.feature_tallies[label]:
                    self.feature_tallies[label][word] = 0
                self.feature_tallies[label][word] += 1

    def classify(self, data):  
        results = []  
        for line in data:
            _, _, content = line.split('|')  
            words = self.preprocess_text(content)
            max_prob = float('-inf')
            max_class = None
            for c in self.categories:
                prob_c = math.log(self.category_counts[c]) - math.log(sum(self.category_counts.values()))
                prob_x_c = 0
                for word in words:
                    count_wc = self.feature_tallies[c].get(word, 0) + 1
                    count_c = sum(self.feature_tallies[c].values()) + len(self.words)
                    prob_wc = math.log(count_wc) - math.log(count_c)
                    prob_x_c += prob_wc
                prob_c_x = prob_c + prob_x_c
                if prob_c_x > max_prob:
                    max_prob = prob_c_x
                    max_class = c
            results.append(max_class)
        return results
