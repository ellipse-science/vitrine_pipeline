# Packages
library(tidyverse, warn.conflicts = FALSE)
library(tube)
library(tokenizers)

# Objet de connexion
condw <- ellipse_connect(env = "PROD", database = "datawarehouse")

# Requête des données
Data_radar <- ellipse_query(condw, "r-media-headlines") |> 
  dplyr::mutate(extraction_date = as.Date(extraction_date)) |>
dplyr::filter(media_id != "CNN" & media_id != "FXN") |>
dplyr::collect()

# Filtrer les médias anglophones
Data_radar_en <- Data_radar |> 
  filter(media_id == c("CBC", "CTV", "GAM", "GN", "MG", "NP", "TTS", "VS"))

#Filtrer les médias francophones
Data_radar_fr <- Data_radar |> 
  filter(media_id == c("JDM", "LAP", "LED", "RCI", "TVA"))

# Séparer les corpus en phrases
# Pour le corpus anglais
Data_radar_en_split <- Data_radar_en %>%
  mutate(body = tokenize_sentences(as.character(body))) %>%  
  unnest_longer(body) %>%                                    
  mutate(body = trimws(body))                                

# Pour le corpus français
Data_radar_fr_split <- Data_radar_fr %>%
  mutate(body = tokenize_sentences(as.character(body))) %>%
  unnest_longer(body) %>%
  mutate(body = trimws(body))

# Sélection du subset aléatoire
set.seed(123)  

Data_subset_en <- Data_radar_en_split |> 
  filter(str_count(body, "\\w+") > 5) |> 
  sample_n(10000)     

Data_subset_en <- Data_subset_en |> 
  select(id, extraction_date, body, media_id) |> 
  mutate(lang = "EN")

Data_subset_fr <- Data_radar_fr_split |> 
  filter(str_count(body, "\\w+") > 5) |> 
  sample_n(10000)                         

Data_subset_fr <- Data_subset_fr |> 
  select(id, extraction_date, body, media_id) |> 
  mutate(lang = "FR")

# Merge subset EN et FR
Data_subset <- bind_rows(Data_subset_en, Data_subset_fr)

# save subset
write_csv(Data_subset, "data/radar_subset.csv")

Test_set <- read_csv("data/radar_subset_test_en_annotated.csv")
