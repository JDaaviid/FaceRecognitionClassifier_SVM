
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA 
from sklearn.pipeline import make_pipeline 
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)

print(faces.images.shape) # (1348 imágenes, 62px altura, 47px anchura)


pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
#150 componentes de las 2914 disponibles
svc = SVC(kernel="rbf", class_weight="balanced")
#kernel="rbf", ya que es más interesante para las imágenes obtener rasgos circulares que los clasifique.
#class_weight="balanced", para que el clasificador pondere los rasgos que son más importantes.
model = make_pipeline(pca, svc) #encadenar instrucciones una tras otra

#Validación del modelo:
from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(faces.data, faces.target, random_state=42)
#Buscamos el mejor modelo:
param_grid = {
    "svc__C":[0.1, 1, 5, 10, 50],
    "svc__gamma":[0.0001, 0.0005, 0.001, 0.005, 0.01] #Valor gamma de la función radial
}
grid = GridSearchCV(model, param_grid)
grid.fit(Xtrain, Ytrain) 

classifier = grid.best_estimator_
yfit = classifier.predict(Xtest)

fig, ax = plt.subplots(8,6, figsize=(16,9))

for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(Xtest[i].reshape(62,47), cmap="bone")
    ax_i.set(Xticks=[], yticks=[])
    ax_i.set_ylabel(faces.target_names[yfit[i]].split()[-1],
    color = "green" if yfit[i]==Ytest[i] else "red")

fig.suptitle("Predicciones de las imágenes (incorrectas en rojo)", size=15)
plt.show()

accuracy = metrics.accuracy_score(Ytest, yfit)

print(accuracy) # Precisión 0.8486646884272997