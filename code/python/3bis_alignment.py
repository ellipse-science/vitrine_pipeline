"""
PROJET : vitrine_pipeline
TITRE : 3bis_alignment.py

OBJECTIF :
-----------
Ce script permet de créer deux fichiers dans le répertoire
data/processed/validation/subvalidation :

1. Un fichier JSONL nommé "alignment.jsonl" qui regroupe l'ensemble des phrases annotées 
   (issues des fichiers JSONL présents dans data/processed/validation/annotated_jsonl)
   MAIS uniquement celles pour lesquelles il existe un désaccord entre les annotateurs.
   Pour chaque phrase, tous les labels annotés sont conservés, chacun étant suffixé par "_" 
   suivi du nom de l'annotateur (dérivé du nom du fichier, par exemple "_antoine").

2. Un fichier de configuration des labels nommé "label_config.json". 
   Ce fichier contient pour chaque variante de label (par exemple, theme_macroeconomics_antoine 
   et theme_macroeconomics_jeremy) un objet qui attribue une couleur de fond et une couleur de texte.
   Ainsi, toutes les variantes représentant le même label de base (extrait en retirant le suffixe)
   se verront attribuer la même couleur.

Auteur :
---------
Antoine Lemor
"""

import os
import json
import glob
import colorsys

# -----------------------------------------------------------------------------
# Fonctions de lecture et de sélection de fichiers
# -----------------------------------------------------------------------------
def lister_fichiers_jsonl(dossier):
    """
    Liste tous les fichiers .jsonl présents dans le dossier spécifié.

    Paramètres
    ----------
    dossier : str
        Chemin vers le dossier où chercher les fichiers .jsonl.

    Retour
    ------
    list
        Liste des chemins complets des fichiers .jsonl trouvés.
    """
    pattern = os.path.join(dossier, "*.jsonl")
    fichiers = glob.glob(pattern)
    return fichiers

def demander_selection_fichiers(liste_fichiers, message_invitation):
    """
    Affiche la liste des fichiers disponibles et demande à l'utilisateur
    de sélectionner un ou plusieurs fichiers par leur indice.

    Paramètres
    ----------
    liste_fichiers : list
        Liste des chemins complets des fichiers.
    message_invitation : str
        Message d'invitation pour la sélection.

    Retour
    ------
    list
        Liste des chemins des fichiers sélectionnés.
    """
    if not liste_fichiers:
        print("Aucun fichier trouvé.")
        return []
    
    print(message_invitation)
    for i, chemin in enumerate(liste_fichiers):
        print(f"{i} -> {chemin}")
    choix = input("Entrez les indices des fichiers à sélectionner, séparés par des virgules : ")
    indices = [x.strip() for x in choix.split(",") if x.strip().isdigit()]
    
    fichiers_selectionnes = []
    for idx in indices:
        idx_int = int(idx)
        if 0 <= idx_int < len(liste_fichiers):
            fichiers_selectionnes.append(liste_fichiers[idx_int])
    return fichiers_selectionnes

# -----------------------------------------------------------------------------
# Lecture des annotations d'un fichier JSONL et suffixage des labels
# -----------------------------------------------------------------------------
def lire_annotations_jsonl(filepath):
    """
    Lit un fichier .jsonl d'annotations et retourne un dictionnaire structuré.

    Pour chaque ligne, on extrait le texte et la liste des labels.
    Chaque label est suffixé par "_" suivi du nom de l'annotateur, qui est
    déduit du nom du fichier (sans extension).

    La fonction retourne un dictionnaire de la forme :
        { texte : { annotateur: set(labels_modifiés) } }

    Paramètres
    ----------
    filepath : str
        Chemin complet du fichier .jsonl.

    Retour
    ------
    dict
        Dictionnaire avec comme clé le texte et comme valeur un sous-dictionnaire
        associant l'annotateur (nom du fichier sans extension) à son ensemble de labels modifiés.
    """
    annotateur = os.path.splitext(os.path.basename(filepath))[0]
    dico = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Erreur de décodage JSON dans le fichier {filepath} : {line}")
                continue
            texte = data.get("text", "").strip()
            labels = data.get("label", [])
            if not isinstance(labels, list):
                labels = []
            # Suffixer chaque label avec "_" + nom de l'annotateur
            labels_modifies = { f"{lbl}_{annotateur}" for lbl in labels }
            # Conserver la structure par phrase et par annotateur
            if texte in dico:
                dico[texte][annotateur] = labels_modifies
            else:
                dico[texte] = { annotateur: labels_modifies }
    return dico

# -----------------------------------------------------------------------------
# Fusion des annotations de plusieurs fichiers (conservation par phrase et annotateur)
# -----------------------------------------------------------------------------
def fusionner_annotations(dicts_list):
    """
    Fusionne plusieurs dictionnaires d'annotations en un seul dictionnaire global.

    Chaque dictionnaire a la forme : { texte : { annotateur: set(labels) } }.
    La fusion se fait par phrase, en combinant les annotations de chaque annotateur.

    Paramètres
    ----------
    dicts_list : list
        Liste de dictionnaires issus de la fonction lire_annotations_jsonl().

    Retour
    ------
    dict
        Dictionnaire fusionné : texte -> { annotateur: set(labels), ... }
    """
    fusion = {}
    for d in dicts_list:
        for texte, ann_dict in d.items():
            if texte not in fusion:
                fusion[texte] = {}
            for ann, labels in ann_dict.items():
                fusion[texte][ann] = labels
    return fusion

# -----------------------------------------------------------------------------
# Filtrage des phrases contenant un désaccord entre annotateurs
# -----------------------------------------------------------------------------
def filter_desaccords(merged_annotations):
    """
    Filtre le dictionnaire fusionné pour ne conserver que les phrases
    présentant un désaccord entre les annotateurs.

    Pour chaque phrase, on extrait l'ensemble des labels de base pour chaque annotateur 
    (en retirant le suffixe, c'est-à-dire la partie après le dernier '_').
    Si tous les annotateurs donnent exactement le même ensemble de labels de base, il y a accord.
    Sinon, il y a désaccord et la phrase est conservée.

    Paramètres
    ----------
    merged_annotations : dict
        Dictionnaire global de la forme : { texte: { annotateur: set(modified_labels), ... } }

    Retour
    ------
    dict
        Sous-ensemble de merged_annotations ne contenant que les phrases avec désaccord.
    """
    filtered = {}
    for texte, ann_dict in merged_annotations.items():
        # Nécessite au moins deux annotateurs pour comparer
        if len(ann_dict) < 2:
            continue
        base_labels_list = []
        for annot, labels in ann_dict.items():
            # Pour chaque label, extraire la partie avant le dernier '_'
            bases = set()
            for label in labels:
                if "_" in label:
                    bases.add(label.rsplit('_', 1)[0])
                else:
                    bases.add(label)
            base_labels_list.append(bases)
        # Si tous les ensembles de labels de base sont identiques, il y a accord
        if all(bases == base_labels_list[0] for bases in base_labels_list):
            continue
        else:
            filtered[texte] = ann_dict
    return filtered

# -----------------------------------------------------------------------------
# Écriture du fichier JSONL d'alignement
# -----------------------------------------------------------------------------
def ecrire_jsonl(annotations_dict, output_filepath):
    """
    Écrit le dictionnaire d'annotations dans un fichier JSONL.

    Pour chaque phrase, on écrit un objet JSON comportant :
      - "text" : le texte original
      - "label" : la liste (triée) de tous les labels annotés (avec suffixe)

    Paramètres
    ----------
    annotations_dict : dict
        Dictionnaire : texte -> { annotateur: set(labels) }.
    output_filepath : str
        Chemin complet du fichier de sortie.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f:
        for texte, ann_dict in annotations_dict.items():
            # Conserver la totalité des labels (union de tous les annotateurs)
            labels_union = set()
            for labels in ann_dict.values():
                labels_union.update(labels)
            objet = {
                "text": texte,
                "label": sorted(list(labels_union))
            }
            f.write(json.dumps(objet, ensure_ascii=False) + "\n")
    print(f"[OK] Fichier alignment.jsonl écrit : {output_filepath}")

# -----------------------------------------------------------------------------
# Génération de couleurs distinctes
# -----------------------------------------------------------------------------
def generate_distinct_colors(n):
    """
    Génère n couleurs distinctes en format hexadécimal.
    Les couleurs sont créées en espaçant uniformément la teinte sur la roue des couleurs.
    
    Paramètres
    ----------
    n : int
        Nombre de couleurs à générer.
        
    Retour
    ------
    list
        Liste de chaînes hexadécimales représentant les couleurs.
    """
    colors = []
    for i in range(n):
        hue = i / n          # valeur de teinte entre 0 et 1
        lightness = 0.5      # luminosité fixée
        saturation = 0.65    # saturation fixée
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#{:02X}{:02X}{:02X}'.format(int(r*255), int(g*255), int(b*255))
        colors.append(hex_color)
    return colors

# -----------------------------------------------------------------------------
# Création du fichier de configuration des labels
# -----------------------------------------------------------------------------
def assign_colors_to_base_labels(base_labels):
    """
    Associe à chaque label de base une couleur de fond unique générée dynamiquement,
    et fixe la couleur du texte à "#ffffff".
    
    Paramètres
    ----------
    base_labels : list
        Liste des labels de base (sans suffixe).

    Retour
    ------
    dict
        Dictionnaire mappant chaque label de base à un tuple (backgroundColor, textColor).
    """
    n = len(base_labels)
    distinct_colors = generate_distinct_colors(n)
    colors = {}
    for base, color in zip(sorted(base_labels), distinct_colors):
        colors[base] = (color, "#ffffff")
    return colors

def creer_config_labels(merged_annotations, output_filepath):
    """
    Crée un fichier de configuration des labels contenant tous les cas de figure
    pour chaque label annoté. Pour chaque variante (par exemple, theme_macroeconomics_antoine,
    theme_macroeconomics_jeremy), on attribue une couleur de fond et une couleur de texte
    identiques si elles représentent le même label de base.

    La structure de chaque objet dans le JSON est :
      {
        "id": <entier unique>,
        "text": "<label_variant>",
        "prefixKey": null,
        "suffixKey": "0",
        "backgroundColor": "<couleur>",
        "textColor": "<couleur>"
      }

    Paramètres
    ----------
    merged_annotations : dict
        Dictionnaire global : texte -> { annotateur: set(labels) }.
    output_filepath : str
        Chemin complet du fichier de sortie (par ex. .../label_config.json).
    """
    # Récupérer l'ensemble de toutes les variantes de labels
    all_variants = set()
    for ann_dict in merged_annotations.values():
        for labels in ann_dict.values():
            all_variants.update(labels)
    
    # Organiser les variantes par label de base (extraction en retirant le suffixe)
    base_to_variants = {}
    for variant in sorted(all_variants):
        if "_" in variant:
            base = variant.rsplit('_', 1)[0]
        else:
            base = variant
        base_to_variants.setdefault(base, []).append(variant)
    
    # Attribution des couleurs par label de base
    base_labels = list(base_to_variants.keys())
    colors = assign_colors_to_base_labels(base_labels)
    
    config_list = []
    current_id = 1000
    for base, variants in base_to_variants.items():
        bg_color, txt_color = colors.get(base, ("#CCCCCC", "#000000"))
        for variant in variants:
            config_item = {
                "id": current_id,
                "text": variant,
                "prefixKey": None,
                "suffixKey": "0",
                "backgroundColor": bg_color,
                "textColor": txt_color
            }
            config_list.append(config_item)
            current_id += 1

    # Écriture du fichier de configuration JSON
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(config_list, f, ensure_ascii=False, indent=2)
    print(f"[OK] Fichier de configuration des labels écrit : {output_filepath}")

# -----------------------------------------------------------------------------
# Fonction principale
# -----------------------------------------------------------------------------
def main():
    """
    Point d'entrée du script.
    
    1) Demande à l'utilisateur de sélectionner les fichiers .jsonl à prendre en compte.
    2) Lit les annotations de chaque fichier en suffixant les labels par le nom de l'annotateur.
    3) Fusionne les annotations en conservant la structure par phrase et par annotateur.
    4) (Optionnel) Demande si l'utilisateur souhaite exclure les annotations déjà présentes dans base.jsonl.
    5) Filtre pour ne conserver que les phrases présentant un désaccord entre annotateurs.
    6) Écrit le fichier "alignment.jsonl" dans le dossier data/processed/validation/subvalidation.
    7) Crée le fichier de configuration des labels "label_config.json" dans le même dossier.
    """
    # Répertoire source des fichiers annotés
    dossier_source = os.path.join("data", "processed", "validation", "annotated_jsonl")
    liste_jsonl = lister_fichiers_jsonl(dossier_source)
    
    # Sélection des fichiers à utiliser
    fichiers_selectionnes = demander_selection_fichiers(
        liste_jsonl,
        "Sélectionnez les fichiers .jsonl à utiliser pour créer le fichier d'alignement :"
    )
    
    if not fichiers_selectionnes:
        print("Aucun fichier sélectionné, le script s'arrête.")
        return
    
    # Lecture des annotations pour chaque fichier sélectionné
    list_of_dicts = []
    for chemin in fichiers_selectionnes:
        print(f"Lecture du fichier : {chemin}")
        dico = lire_annotations_jsonl(chemin)
        list_of_dicts.append(dico)
    
    # Fusion des annotations (structure : texte -> { annotateur: set(labels) })
    merged_annotations = fusionner_annotations(list_of_dicts)
    
    # Nouvelle fonctionnalité : demander s'il faut exclure les phrases déjà annotées dans base.jsonl
    base_jsonl_path = os.path.join("data", "processed", "validation", "annotated_jsonl", "base.jsonl")
    base_texts = set()
    if os.path.exists(base_jsonl_path):
        with open(base_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    texte = data.get("text", "").strip()
                    if texte:
                        base_texts.add(texte)
                except json.JSONDecodeError:
                    continue
    reponse_exclusion = input("Voulez-vous exclure les annotations contenues dans 'base.jsonl' de la comparaison entre annotateurs ? (o/n) : ").strip().lower()
    if reponse_exclusion in ['o', 'oui'] and base_texts:
        nb_avant = len(merged_annotations)
        merged_annotations = {texte: ann_dict for texte, ann_dict in merged_annotations.items() if texte not in base_texts}
        nb_exclus = nb_avant - len(merged_annotations)
        print(f"Exclusion effectuée : {nb_exclus} phrases exclues sur {nb_avant}.")
    else:
        print("Aucune exclusion effectuée.")
    
    # Filtrer pour ne conserver que les phrases avec désaccord entre annotateurs
    filtered_annotations = filter_desaccords(merged_annotations)
    
    if not filtered_annotations:
        print("Aucune phrase avec désaccord trouvée. Aucun fichier alignment.jsonl ne sera généré.")
    else:
        # Chemin de sortie du fichier alignment.jsonl
        output_alignment = os.path.join("data", "processed", "validation", "subvalidation", "alignment.jsonl")
        ecrire_jsonl(filtered_annotations, output_alignment)
    
    # Création du fichier de configuration des labels (sur l'ensemble des annotations)
    output_config = os.path.join("data", "processed", "validation", "subvalidation", "label_config.json")
    creer_config_labels(merged_annotations, output_config)

if __name__ == "__main__":
    main()