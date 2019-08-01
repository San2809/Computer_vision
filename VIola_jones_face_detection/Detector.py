import numpy as np
import math
import pickle
from sklearn.feature_selection import SelectPercentile, f_classif

class Detector:
    def __init__(self, T = 10):

        self.T = T
        self.alphas = []
        self.clfs = []

    def training(self, train, posti, negat):

        wgts = np.zeros(len(train))
        data = []
        for x in range(len(train)):
            data.append((int_img(train[x][0]), train[x][1]))
            if train[x][1] == 1:
                wgts[x] = 1.0 / (2 * posti)
            else:
                wgts[x] = 1.0 / (2 * negat)

        features = self.feat_build(data[0][0].shape)
        X, y = self.apply_features(features, data)
        indices = SelectPercentile(f_classif, percentile=10).fit(X.T, y).get_support(indices=True)
        X = X[indices]
        features = features[indices]

        for t in range(self.T):
            wgts = wgts / np.linalg.norm(wgts)
            weak_classifiers = self.weak_clfy(X, y, features, wgts)
            clf, error, accuracy = self.select_best(weak_classifiers, wgts, data)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                wgts[i] = wgts[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0/beta)
            self.alphas.append(alpha)
            self.clfs.append(clf)


    def weak_clfy(self, X, y, features, wgts):

        total_pos, total_neg = 0, 0
        for w, label in zip(wgts, y):
            if label == 1:
                total_pos += w
            else:
                total_neg += w

        clsfr = []
        
        for index, feature in enumerate(X):

            
            applied_feature = sorted(zip(wgts, feature, y), key=lambda x: x[1])

            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            for w, f, label in applied_feature:
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = features[index]
                    best_threshold = f
                    best_polarity = 1 if pos_seen > neg_seen else -1

                if label == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w
            
            clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
            clsfr.append(clf)
        return clsfr
                
    def feat_build(self, image_shape):

        height, width = image_shape
        features = []
        for w in range(1, width+1):
            for h in range(1, height+1):
                i = 0
                while i + w < width:
                    j = 0
                    while j + h < height:
                        #2 rectangle features
                        immediate = RectangleRegion(i, j, w, h)
                        right = RectangleRegion(i+w, j, w, h)
                        if i + 2 * w < width: #Horizontally Adjacent
                            features.append(([right], [immediate]))

                        bottom = RectangleRegion(i, j+h, w, h)
                        if j + 2 * h < height: #Vertically Adjacent
                            features.append(([immediate], [bottom]))
                        
                        right_2 = RectangleRegion(i+2*w, j, w, h)
                        #3 rectangle features
                        if i + 3 * w < width: #Horizontally Adjacent
                            features.append(([right], [right_2, immediate]))

                        bottom_2 = RectangleRegion(i, j+2*h, w, h)
                        if j + 3 * h < height: #Vertically Adjacent
                            features.append(([bottom], [bottom_2, immediate]))

                        #4 rectangle features
                        bottom_right = RectangleRegion(i+w, j+h, w, h)
                        if i + 2 * w < width and j + 2 * h < height:
                            features.append(([right, bottom], [immediate, bottom_right]))

                        j += 1
                    i += 1
        return np.array(features)

    def select_best(self, clsfr, wgts, data):

        best_clf, best_error, best_accuracy = None, float('inf'), None
        for clf in clsfr:
            error, accuracy = 0, []
            for data, w in zip(data, wgts):
                correctness = abs(clf.clfy(data[0]) - data[1])
                accuracy.append(correctness)
                error += w * correctness
            error = error / len(data)
            if error < best_error:
                best_clf, best_error, best_accuracy = clf, error, accuracy
        return best_clf, best_error, best_accuracy
    
    def apply_features(self, features, data):

        X = np.zeros((len(features), len(data)))
        y = np.array(list(map(lambda data: data[1], data)))
        i = 0
        for positiv, negati in features:
            feature = lambda intgimg: sum([pos.featcomp(intgimg) for pos in positiv]) - sum([neg.featcomp(intgimg) for neg in negati])
            X[i] = list(map(lambda data: feature(data[0]), data))
            i += 1
        return X, y

    def clfy(self, image):

        total = 0
        intgimg = int_img(image)
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.clfy(intgimg)
        return 1 if total >= 0.5 * sum(self.alphas) else 0

    def save(self, filename):

        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):

        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)

class WeakClassifier:
    def __init__(self, positiv, negati, threshold, polarity):

        self.positiv = positiv
        self.negati = negati
        self.threshold = threshold
        self.polarity = polarity
    
    def clfy(self, x):

        feature = lambda intgimg: sum([pos.featcomp(intgimg) for pos in self.positiv]) - sum([neg.featcomp(intgimg) for neg in self.negati])
        return 1 if self.polarity * feature(x) < self.polarity * self.threshold else 0
    

class RectangleRegion:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def featcomp(self, intgimg):

        return intgimg[self.y+self.height][self.x+self.width] + intgimg[self.y][self.x] - (intgimg[self.y+self.height][self.x]+intgimg[self.y][self.x+self.width])

        
def int_img(image):

    inimg = np.zeros(image.shape)
    p = np.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[y])):
            p[y][x] = p[y-1][x] + image[y][x] if y-1 >= 0 else image[y][x]
            inimg[y][x] = inimg[y][x-1]+p[y][x] if x-1 >= 0 else p[y][x]
    return inimg