# Example 2: Clustering the IRIS Dataset
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import math
from pandas import crosstab

meas = load_iris().data

X = meas[:, [2, 3]]
Y = load_iris().target
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
        #Παρουσιάζει την κλάση που ανήκει η κάθε παρατήρηση και τις πραγματικές κλάσεις
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
        ax1.set_title('K-Means')
        for i in range(numberOfClusters):
            ax1.plot(X[IDX==i][:,0], X[IDX==i][:,1], marker='o', linewidth=0, label = i+1)
            for j in range(numberOfRows):
                if IDX[j] == i:
                    sse = sse + math.dist(X[j], C[i])**2
        ax1.scatter(C[:,0], C[:,1], marker='x', color='black', s=100, linewidth=1, label="Centroids", zorder=10)
        ax2.set_title("Original")
        ax2.scatter(X[:,0],X[:,1],c=Y, cmap='brg')

        print('\n\nΚατανομή σε συστάδες σε σχέση με τις πραγματικ΄ς κλάσεις (ισχύει μόνο για 3 συστάδες): \n',crosstab(IDX, Y))

        print("\n\nsse = %.3f" % sse)
        if numberOfClusters < numberOfRows:
            print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, IDX))

        print("\n\nΤέλος προγράμματος")
        break;
