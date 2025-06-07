# Compression de Réseaux de Neurones : Pénalisation L1 vs Contrainte L0

Ce projet compare deux méthodes de compression de réseaux de neurones, la **pénalisation L1** et la **contrainte L0**, dans une perspective d'efficacité ("Green AI").

L'expérimentation évalue le compromis entre la précision d'un modèle MLP sur MNIST et ses coûts computationnels (latence, énergie, taille mémoire).

## Principaux Résultats

-   **Efficacité** : La méthode par **contrainte L0 (IHT)** est nettement supérieure. Elle atteint une sparsité élevée (plus de 20%) avec une perte de performance négligeable.

-   **Limites** : La **pénalisation L1 (ISTA)**, bien que robuste, génère une sparsité trop faible (moins de 0.2%) pour offrir des gains pratiques significatifs.

-   **Gains "Green AI"** : Avec la contrainte L0, nous observons une réduction de la latence jusqu'à **35%** et de la consommation énergétique jusqu'à **37%**.

-   **Taille du Modèle** : La méthode L0 permet une réduction notable de l'empreinte mémoire, un atout majeur pour les systèmes embarqués ("Edge AI").

## Structure du Projet

-   `config.py`: Gère les hyperparamètres des expériences.
-   `dataset.py`: S'occupe du chargement des données MNIST.
-   `model.py`: Définit l'architecture du réseau de neurones (MLP).
-   `utils.py`: Contient les fonctions d'évaluation (précision, sparsité, métriques Green AI).
-   `train_pl1.py`: Script d'entraînement pour la méthode de pénalisation L1.
-   `train_pl0.py`: Script d'entraînement pour la méthode de contrainte L0.
-   `plot_results.py`: Génère les graphiques et l'analyse comparative des résultats.

## Prérequis

Pour installer les dépendances nécessaires, exécutez la commande suivante à la racine du projet :

```bash
pip install -r requirements.txt
```

## Instructions d'Exécution

1.  **Entraînement avec Pénalisation L1**
    ```bash
    python train_pl1.py
    ```
    Cette commande génère le fichier `results_PL1_stats.pkl`.

2.  **Entraînement avec Contrainte L0**
    ```bash
    python train_pl0.py
    ```
    Cette commande génère le fichier `results_PL0_stats.pkl`.

3.  **Génération des Graphiques**
    ```bash
    python plot_results.py
    ```
    Ce script utilise les fichiers `.pkl` pour sauvegarder les figures comparatives et afficher une analyse dans le terminal.

## Méthodologie

Nous comparons deux approches d'optimisation :

1.  **Pénalisation L1 (Convexe)** :
    Minimisation de la fonction de coût pénalisée par la norme L1 des poids, résolue par ISTA.
    $$ \min_{w} \mathcal{L}(w)+\lambda||w||_{1} $$

2.  **Contrainte L0 (Non-convexe)** :
    Minimisation de la fonction de coût en contraignant le nombre de poids non nuls, résolue par IHT.
    $$ \min_{w} \mathcal{L}(w) \quad \text{s.c.} \quad ||w||_{0}\le k $$

## Auteurs

- Hamza OUZAHRA
- Ahmed TERRAF
- Samia TOUILE
- Marwen JADLAQUI
