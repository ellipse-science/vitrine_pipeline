# Vitrine_pipeline

## Description

Ce dépôt vise à :

1. Créer un sous-ensemble de données à partir des extraits d’articles issus de **Radar Plus afin de tester, de comparer et de valider le dictionnaire de catégorisation actuel (12 catégories)** des annotations de LLMs locaux, un processus d'apprentissage machine basé sur [Do et al. (2022)](https://journals.sagepub.com/doi/full/10.1177/00491241221134526), et des annotations manuelles.
2. **Utiliser un outil générique d’annotation par LLM (`llm_annotation_tool.py`) pour annoter n’importe quelle base de données SQL ou fichier CSV de façon propre est structurée** (chaque sortie est nettoyée et structurée en JSON). Il est possible de chanegr le modèle appelé (par n'importe quel modèle disponible avec Ollama), personnaliser les prompts (tant qu'il réponde à la structure attendue par le code), et configurer d’autres paramètres d’annotation.
3. En bout de ligne **aboutir à un outil d'annotation complètement automatisé** en un seul pipeline : prompt -> annotations par LLMs locaux (subset) -> entrainement de modèles BERT -> annotation complète

Ce README est divisé en deux grandes parties principales :

1. **Test des 12 catégories Radar Plus**  
   - Test du dictionnaire permettant de catégoriser les phrases en 12 catégories.
   - Comparaison avec des annotations manuelles + LLMs locaux + apprentissage machin (Bert)

2. **Présentation de `llm_annotation_tool.py`**  
   - Un code permettant de faire de l’annotation LLM sur toute base de données (csv, SQL, etc.)
   - Description de ses fonctionnalités

---

## Table des matières
1. [Dépendances et environnement](#dépendances-et-environnement)
2. [Structure du rojet](#structure-du-projet)
3. [Installation et configuration](#installation-et-configuration)
4. [Utilisation](#utilisation)
   - [Partie 1 : Test et validation des 12 catégories (dictionnaire, LLM, apprentissage automatique, annotations manuelles).](#partie-1--test-et-validation-des-12-catégories-dictionnaire-llm-apprentissage-automatique-annotations-manuelles)
   - [Partie 2 : Outil d'annotation avec llm_annotation_tool.py](#partie-2--outil-dannotation-avec-llm_annotation_toolpy)



---

## Dépendances et environnement

Les principales bibliothèques (Python) requises sont :
- **pandas**  
- **random**  
- **os**  
- **sys**  
- **tqdm**  
- **ollama** (pour interagir avec un modèle local via Ollama)  
- **sqlalchemy**  
- **json**, **re**, **math**  
- **concurrent.futures**  
- **logging**

En R (pour le script `radar_extraction.R`) :
- **tidyverse**  
- **tube** (pour des opérations possibles sur YouTube, même si peu utilisé dans cet exemple)  
- **tokenizers**

Pour installer les dépendances Python, vous pouvez utiliser le fichier `requirements.txt` situé à la racine du projet :

```bash
pip install -r requirements.txt
```

---

## Structure du projet

Voici la structure simplifiée du dépôt :

```plaintext
vitrine_pipeline
├── README.md
├── code
│   ├── python
│   │   ├── 1_subset_test_creation.py
│   │   ├── 2_JSONL.py
│   │   └── llm_annotation_tool.py
│   └── r
│       └── radar_extraction.R
├── data
│   ├── processed
│   │   ├── subset
│   │   │   ├── radar_subset_test.csv
│   │   │   ├── radar_subset_test_annotated_100.csv
│   │   └── validation
│   │       └── (JSONL générés par `2_JSONL.py`, ex. annotator_1.jsonl, etc.)
│   └── raw
│       ├── prompt
│       │   └── prompt.txt
│       └── subset
│           └── radar_subset.csv
└── requirements.txt
```

- **`code/r/radar_extraction.R`**  
  - Script R pour extraire des données issues de *radar plus*, filtrer par langue, séparer en phrases, et générer des CSV (par ex. `radar_subset.csv`).

- **`code/python/1_subset_test_creation.py`**  
  - Script python sélectionnant aléatoirement 100 phrases du csv source contenant un échantillon de 20 000 phrases (`radar_subset_en.csv`) pour générer un fichier test (`radar_subset_test.csv`).

- **`code/python/2_JSONL.py`**  
Pour créer les fichiers destinés à Doccano (JSONL).  
  - Assure une répartition 50 % EN / 50 % FR
  - Génère 20 % de phrases communes pour tous les annotateurs et 80 % de phrases uniques
  - Crée un fichier de configuration Doccano (`doccano_config.json`)  
  - Produit un fichier JSONL par annotateur (ex. `annotator_1.jsonl`, `annotator_2.jsonl`, etc.)  
  - Affiche un résumé statistique (répartition par langue, par label, etc.)

- **`code/python/llm_annotation_tool.py`**  
  - Script Python générique pour annoter du texte avec un LLM.  
  - Peut se connecter à PostgreSQL ou lire/écrire un CSV.  
  - Nettoie et normalise la sortie LLM sous forme de JSON, puis l’insère dans la même base ou CSV.

- **`data/raw/subset/radar_subset.csv`**  
  - Données brutes contenant 20 000 phrases extraites de la base radar.

- **`data/processed/subset/radar_subset_test.csv`**  
  - Sous-ensemble test de 100 phrases, créé par `llm_annotation_tool.py` et DeepSeek-R1.

- **`data/processed/subset/radar_subset_test_100.csv`**  
  - Sous-ensemble test de 100 phrases annotées par `1_subset_test_creation.py`.

- **`data/raw/prompt/prompt.txt`**  
  - Fichier texte contenant le prompt et la section `**Clés JSON Attendues**`.  
  - Ce prompt est utilisé par `llm_annotation_tool.py`.

- **`requirements.txt`**  
  - Liste des dépendances Python principales.

---

## Installation et Configuration

1. **Cloner le dépôt**  
   ```bash
   git clone https://github.com/username/vitrine_pipeline.git
   cd vitrine_pipeline
   ```

2. **Installer les dépendances Python**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurer Ollama pour l’utilisation d’un modèle local**  
   - Suivre les instructions d’[Ollama](https://github.com/jmorganca/ollama) pour installer le modèle (ex. `deepseek-r1:70b`).  

---

## Utilisation

### Partie 1 : Test et validation des 12 catégories (dictionnaire, LLM, apprentissage automatique, annotations manuelles)

#### 1.1. Extraire un grand sous-ensemble (20 000 phrases)

- **But** : générer un fichier `radar_subset.csv` de 20 000 phrases (10 000 anglophones + 10 000 francophones) depuis *radar plus*.  
- **Script** : `code/r/radar_extraction.R`.  
  ```r
  # Dans R:
  source("code/r/radar_extraction.R")
  ```

#### 1.2. Créer un sous-ensemble de test (100 phrases)

- **But** : obtenir un échantillon restreint pour tester le dictionnaire et comparer avec des LLMs locaux + annotations manuelles 
- **Script** : `code/python/1_subset_test_creation.py`.  
  ```bash
  python code/python/1_subset_test_creation.py
  ```

#### 1.3. Générer des JSONL pour Doccano (50% EN, 50% FR)

- **But** : préparer les fichiers JSONL (et un fichier de config) pour Doccano (ou un autre outil d’annotation manuelle). La composition des JSONL est de 80% d'annotations différentes, et 20% de communes si plus de de 2 annotateurs sélectionnés. 
- **Script** : `code/python/2_JSONL.py`.  
  ```bash
  python code/python/2_JSONL.py
  ```

---

### Partie 2 : Outil d’annotation avec `llm_annotation_tool.py`

Le script `llm_annotation_tool.py` est un outil qui permet de faire de façon générique de l’annotation via un LLM local (via [Ollama](https://github.com/jmorganca/ollama)) sur plusieurs formats de données :  
- **CSV** (testé et fonctionnel)  
- **PostgreSQL** (testé et fonctionnel)  
- **Excel**, **Parquet**, **RData** ou **RDS** (implémentés, mais pas testés à ce stade)  

Toutes les sorties de LLM sont automatiquement nettoyées et converties en **JSON strict** grâce à des règles de validation internes **basées sur un format strict de prompt**.

---

#### 1) Principales fonctionnalités

**Le script est interactif et vous permet de rentrer directement tout ce qu'il vous faut pour annoter un dataframe à partir d'un prompt et d'un dataframe contenant des données textuelles (tokénisées au niveau le phrase, ce qui est conseillé, ou au minimum où chaque ligne est limitée au *max_tokens* du modèle que vous souhaitez utiliser).**

1. **Lecture multi-format**  
   - Le script peut lire un **CSV**, un fichier **Excel**, un **Parquet**, un **RData** / **RDS** ou se connecter à une **base PostgreSQL**.  
   - Lorsque le script démarre, vous choisissez le format et indiquez le chemin ou les identifiants de connexion SQL.

2. **Création et gestion des colonnes d’annotation**  
   - Vous pouvez rentrer, si besoin, le nom d'une colonne (ex. `annotations_llm`) pour stocker les **JSON** produits.  
   - Possibilité aussi de créer automatiquement une colonne d’**identifiant unique** si vous n’en avez pas déjà.

3. **Saisie interactive des paramètres**  
   À l’exécution, le script vous demande :  
   - **Le format de la source** (CSV, PostgreSQL, Excel, Parquet, RData/RDS).  
   - **Le chemin du fichier** ou les **paramètres PostgreSQL** (hôte, port, base, user, mot de passe, table).  
   - **La colonne texte** (où se trouvent les phrases à annoter).  
   - **La colonne d’identifiant unique** (ou création automatique via `<text_column>_id_for_llm`).  
   - **Le nom de la colonne où stocker l’annotation** (ex. `annotation_llm`).  
   - **Le nombre d’exemples à annoter** (avec possibilité de sélectionner aléatoirement un sous-ensemble).  
   - **Le chemin vers le prompt** (ex. `data/raw/prompt/prompt.txt`).  
   - **Les paramètres du modèle** (temperature, seed, top_p, num_predict, etc.) ou standards si vous voulez les laisser tel quel (appuyer sur entrée). 
   - **Le nombre de processus** (pour le multiprocessing si vous voulez runner deux processus ou plus en même temps, mais attention de pas surcharger la machine).

4. **Mise à jour en temps réel** ce qui signifie que la dataframe annoté est accessible dès la première annotation et mis à jour à chaque annotation 
   - Après chaque annotation, la sortie du modèle est convertie en JSON strict (selon les clés **explicitement définies** dans le prompt) et sauvegardée dans le dataframe (puis réécriture du fichier si en mode fichier).  

5. **Nettoyage des réponses LLM**  
   - Le script réessaye jusqu’à 5 fois si la réponse n’est pas du JSON strict (ex. si le modèle renvoie du texte supplémentaire).  
   - Il retire les balises de code (\`\`\`), les commentaires, et ne garde que les clés attendues.  
   - Vous obtenez donc un JSON final fiable dans la colonne d’annotation.

6. **Multiprocessing**  
   - Le script peut être lancé avec un nombre variable de processus pour accélérer l’annotation sur un gros volume de données.  

7. **Sauvegarde ligne par ligne**  
   - En mode fichier (CSV, Excel, etc.), chaque annotation est immédiatement sauvegardée dans le fichier pour éviter toute perte.  
   - En mode base de données (PostgreSQL), l’update est fait ligne par ligne pour chaque identifiant.

---

#### 2) Formatage requis du prompt

Pour que les **sorties soient bien nettoyées**, vous devez fournir un prompt **qui définit clairement** la portion de texte à analyser et les **clés JSON attendues**. Le script recherche un séparateur **qui doit être placé à la toute fin des instructions** :  
```
**Clés JSON Attendues**
```
et ensuite **identifie les clés** entre guillemets (`"key_name"`).  
- L’exemple type d’un prompt dans `prompt.txt` :  

  ```txt
  Tu es un annotateur de texte. Analyse et résume les principales thématiques abordées dans ces extraits d'une phrase provenant d'articles médiatiques en utilisant les catégories suivantes pour structurer la sortie en JSON. Tu dois écrire exclusivement le JSON sans autre texte explicatif. 
  
  Les catégories doivent être claires et les valeurs appropriées doivent être utilisées :
  
  **Clés JSON d'annotations :**
  - "themes" : "law_and_crime" si la phrase s'apparente à une thématique de loi, de crime ou d'ordre public, "culture_and_nationalism" si la phrase s'apparente à une thématique d'arts, de culture, d'identité ou de nationalisme, "public_lands_and_agriculture" si la phrase s'apparente à une thématique de régions, d'agriculture, de terres publiques, de gestion de la terre et des eaux, de pêche ou de forêt, "governments_and_governance" si la phrase s'apparente à une thématique de gouvernement, de démocratie, d'opérations gouvernementales, d'affaires provinciales et locales, de relations intergouvernementales ou d'union nationale constitutionnelle, "immigration" si la phrase s'apparente à une thématique d'immigration, "rights_liberties_minorities_discrimination" si la phrase s'apparente à une thématique de minorités, de droits civils, de religion ou d'affaires autochtones, "health_and_social_services" si la phrase s'apparente à une thématique de santé, de santé publique ou de services sociaux, "economy_and_labour" si la phrase s'apparente à une thématique d'économie, d'employabilité, de macro-économie, de travail, de commerce extérieur, de commerce intérieur, de logement, de finances publiques ou de transport, "education" si la phrase s'apparente à une thématique d'éducation ou de recherche, "environment_and_energy" si la phrase s'apparente à une thématique d'environnement, d'énergie ou de lutte aux changements climatiques, "international_affairs_and_defense" si la phrase s'apparente à une thématique d'affaires internationales ou de défense, "technology" si la phrase s'apparente à une thématique de technologie, "null" si la phrase ne s'apparente explicitement à aucun de ces thèmes. 
  
  **Instructions :**
  - Suivre strictement la structure des clés définies ci-dessus.
  - Assurer que toutes les clés sont présentes dans le JSON, en utilisant `null` lorsque nécessaire.
  - Ne pas inclure de clés non définies dans la liste ci-dessus.
  - Écrire exclusivement le JSON sans autre commentaire ou explication.
  - Indiquer plusieurs thèmes si plusieurs thèmes sont présents.
  
  **Exemple d'annotation pour le titre :**
  
  During a Liberal cabinet retreat in Halifax last August, Prime Minister Justin Trudeau signalled a possible reduction in permanent resident levels, a major policy reversal for the federal government.
  
  **Exemple de JSON :**
  
  {
  "themes": ["immigration", "governments_and_governance"]
  }
  
  Suivre cette structure pour chaque phrases analysées. Aucun autre commentaire ou détails supplémentaires autre que la structure en JSON demandée et les catégories spécifiées ne doit être rajouté.
  
  **Clés JSON Attendues**
  
  {
  "themes": "",
  }
  ```

- Le code va ainsi extraire automatiquement `["theme"]` comme **clé attendue**.  
- **Tout ce qui précède la section `**Clés JSON Attendues**` est considéré comme le message principal à donner au LLM (consignes, contexte, etc.).**

---

#### 3) Choix du modèle Ollama

Le script commence par énumérer les **modèles Ollama** disponibles sur la machine (ex. `deepseek-r1:70b`, `llama2-7b`, etc.). Vous sélectionnez ensuite celui à utiliser (via un simple menu). Si vous voulez un autre modèle :
- Installez-le dans Ollama.
- Relancez le script pour le voir apparaître.

Si vous voulez changer le modèle **directement dans le code**, cherchez la partie où est créé l’argument `model_name` pour la fonction `process_comment()` et remplacez `model_name` par un nom en dur.

---

#### 4) Exemple de déroulement 

1. **Lancement**  
   ```bash
   python code/python/llm_annotation_tool.py
   ```
2. **Sélection du format**  
   Le script vous demande si vos données sont en CSV, PostgreSQL, Excel, Parquet, RData/rds…  
3. **Sélection du modèle**  
   Le script affiche la liste des modèles Ollama trouvés (via `ollama list`) et vous invite à choisir un numéro.  
4. **Paramétrage**  
   - Le script vous demande de saisir le chemin du fichier (ex. `data/processed/subset/radar_subset_test.csv`) ou les infos de base de données.  
   - Vous indiquez la colonne texte, la colonne d’ID et la colonne d’annotation à créer (ou à réutiliser).  
   - Vous fixez le nombre de lignes à annoter, la méthode de sélection (aléatoire ou non), etc.  
   - Vous fournissez le fichier prompt. Le script détecte et vous montre les clés attendues.  
   - Vous entrez les paramètres du modèle (température, seed, etc.).  
5. **Exécution**  
   Le script envoie chaque phrase au LLM, récupère la sortie, la nettoie, et l’enregistre immédiatement en JSON.  
6. **Vérification**  
   - Toutes les X annotations, il affiche un exemple de JSON validé pour montrer la forme finale.  
   - S’il échoue à obtenir un JSON correct après 5 tentatives, il passe à la phrase suivante (et vous en informe).