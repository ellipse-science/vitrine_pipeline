"""
PROJET : vitrine_pipeline

TITRE : 2_doccano_label_config.py

OBJECTIF :
----------
Ce script génère un fichier JSON de configuration pour Doccano,
avec quatre grands types de labels et des raccourcis clavier uniques.

CARACTÉRISTIQUES :
------------------
1) 4 grandes catégories de labels (avec préfixes différents) :
   - dimensions thématiques (21)
   - thèmes spécifiques (3)
   - dimensions référentielles (11)
   - dimensions sentimentales (3)

2) Chaque label se voit attribuer un raccourci issu de la plage "0-9" puis "a-z".
   Comme il y a 38 labels et 36 raccourcis possibles, les deux derniers n'en auront pas.

3) Chaque grand type a une couleur unique (exemple : tous les "theme_" auront la même couleur).

4) Le fichier JSON (doccano_label_config.json) est écrit dans "data/processed/validation".

Auteur :
---------
Antoine Lemor
"""

import os
import json

def main():
    # ---------------------------------------------------------------------
    # 1) Définition des couleurs pour chaque grande catégorie
    # ---------------------------------------------------------------------
    color_theme = "#FA8072"      
    color_specific = "#87CEFA"     
    color_referential = "#90EE90"  
    color_sentiment = "#FFFACD"    

    # ---------------------------------------------------------------------
    # 2) Définition des labels pour chaque grande catégorie (nom interne)
    # ---------------------------------------------------------------------
    thematics = [
        "law_and_crime",
        "immigration",
        "technology",
        "macroeconomics",
        "labor",
        "transportation",
        "housing",
        "domestic_commerce",
        "foreign_trade",
        "public_lands",
        "agriculture",
        "environment",
        "energy",
        "international_affairs",
        "defense",
        "governments_governance",
        "culture_nationalism",
        "rights_liberties_minorities_discrimination",
        "education",
        "health",
        "social_welfare"
    ]

    specific_themes = [
        "welfare_state",
        "public_finance",
        "early_learning_childcare"
    ]

    referential_parties = [
        "LPC",   # Liberal Party of Canada
        "CPC",   # Conservative Party of Canada
        "BQ",    # Bloc Québécois
        "NDP",   # New Democratic Party
        "GPC",   # Green Party of Canada
        "PPC",   # People's Party of Canada
        "CAQ",   # Coalition Avenir Québec
        "PLQ",   # Parti libéral du Québec
        "PQ",    # Parti Québécois
        "QS",    # Québec Solidaire
        "PCQ"    # Parti Conservateur du Québec
    ]

    sentiment = [
        "positive",
        "negative",
        "neutral"
    ]

    # ---------------------------------------------------------------------
    # 3) Définir la liste des raccourcis disponibles (uniques, de "0" à "9" puis "a" à "z")
    # ---------------------------------------------------------------------
    available_keys = list("0123456789abcdefghijklmnopqrstuvwxyz") 

    # ---------------------------------------------------------------------
    # 4) Création de la configuration des labels
    # ---------------------------------------------------------------------
    labels_config = []
    base_id = 1000  # Identifiant de départ

    def assign_label(text, bg_color):
        nonlocal base_id, available_keys
        # Attribuer un raccourci s'il en reste, sinon None
        suffix_key = available_keys.pop(0) if available_keys else None
        label = {
            "id": base_id,
            "text": text,
            "prefixKey": None,
            "suffixKey": suffix_key,
            "backgroundColor": bg_color,
            "textColor": "#ffffff"
        }
        base_id += 1
        return label

    # Ajouter les labels des différentes catégories dans l'ordre
    for name in thematics:
        full_text = f"theme_{name}"
        labels_config.append(assign_label(full_text, color_theme))

    for name in specific_themes:
        full_text = f"specific_themes_{name}"
        labels_config.append(assign_label(full_text, color_specific))

    for name in referential_parties:
        full_text = f"political_parties_{name}"
        labels_config.append(assign_label(full_text, color_referential))

    for name in sentiment:
        full_text = f"sentiment_{name}"
        labels_config.append(assign_label(full_text, color_sentiment))

    # ---------------------------------------------------------------------
    # 5) Écriture du fichier JSON dans "data/processed/validation"
    # ---------------------------------------------------------------------
    output_dir = os.path.join("data", "processed", "validation")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "doccano_label_config.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(labels_config, f, indent=2, ensure_ascii=False)

    print(f"Fichier de configuration créé : {output_file}")
    print("Vous pouvez l'importer dans Doccano pour configurer vos labels.")

if __name__ == "__main__":
    main()
