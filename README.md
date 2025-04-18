# Detecting-and-rating-humour-and-offense
📁 Structure du projet
Ce projet est composé de trois fichiers Jupyter Notebook :

data_preprocessing.ipynb
Ce notebook contient :

Le prétraitement des données textuelles : nettoyage, standardisation des blagues, suppression du bruit.

L’annotation des blagues avec des labels d’humour et d’offensivité.

La visualisation des distributions de classes.

L’équilibrage des classes pour les tâches de classification.

⚠️ Étant donné que l'exécution de ce fichier peut prendre beaucoup de temps, le fichier CSV généré (dataset_annotated_balanced.csv) est déjà fourni dans le dossier data.

task_one_classification.ipynb
Implémente la classification des blagues comme offensantes ou non, à l’aide de différents modèles tels que SVM, RNN, LSTM et BERT.

task_two_regression.ipynb
Permet d’attribuer un score d’offensivité aux blagues offensantes (sur une échelle de 0 à 5) à l’aide de modèles de régression (RNN, LSTM, BERT).

🚀 Instructions d’exécution
L’ordre d’exécution recommandé est le suivant :

Exécuter task_one_classification.ipynb pour effectuer la classification binaire des blagues.

Une fois la classification effectuée, exécuter task_two_regression.ipynb pour prédire un score d’offensivité sur les blagues offensantes.

📝 Le fichier data_preprocessing.ipynb est optionnel si vous utilisez directement le fichier dataset_annotated_balanced.csv déjà fourni.
