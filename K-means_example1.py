# Example 1
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import math

X=np.array([[0,0],[0,2],[4,0],[4,2]]);
#X=np.array([[0.40,0.53],[0.22,0.38],[0.35,0.32],[0.26,0.19],[0.08,0.41],[0.45,0.3]]);
numberOfRows, numberOfColumns = X.shape
print('Size of the dataset: ', end="")
print(numberOfRows)

while True:
    numberOfClusters = int(input(f"Δώσε τον αριθμό των συστάδων για τον K-means (από 2 έως {numberOfRows}): "))
    if numberOfClusters <= 1 or numberOfClusters > numberOfRows:
        print("Εσφαλμένη τιμή!!\n")
        continue
    else:
        kmeans = KMeans(n_clusters=numberOfClusters).fit(X) #Εφαρμόστε τον αλγόριθμο k-means
        IDX = kmeans.labels_
        C = kmeans.cluster_centers_
        #Εξ ορισμού, ο αλγόριθμος συσταδοποίησης k-means βασίζεται
        #στην ευκλείδεια απόσταση, συνεπώς η Python δεν επιτρέπει τη χρήση
        #άλλων αποστάσεων
        
        sse = 0.0
        plt.figure(1)
        #Παρουσιάζει την κλάση που ανήκει η κάθε παρατήρηση
        for i in range(numberOfClusters):
            plt.plot(X[IDX==i][:,0], X[IDX==i][:,1], marker='o', linewidth=0, label = i+1)
            for j in range(numberOfRows):
                if IDX[j] == i:
                    sse = sse + math.dist(X[j], C[i])**2
        plt.scatter(C[:,0], C[:,1], marker='x', color='black', s=100, linewidth=1, label="Centroids", zorder=10)
        plt.legend()
        plt.show()

        print("\n\nsse = %.3f" % sse)
        if numberOfClusters < numberOfRows:
            print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, IDX))
        
        print("\n\nΤέλος προγράμματος")
        break;
