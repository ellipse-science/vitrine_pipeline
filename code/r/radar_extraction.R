# Packages
library(tidyverse, warn.conflicts = FALSE)
library(tube)

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

# Séparer le corpus en phrases
pattern <- "(?<!\\bM)\\.\\s+(?=[A-Z])|\\.\\s*$"

Data_radar_en_split <- Data_radar_en |>
  mutate(body = strsplit(as.character(body), pattern, perl = TRUE)) |>
  unnest_longer(col = body) |>
  mutate(body = trimws(body))

# Sélection du subset aléatoire
set.seed(123)  

Data_subset <- Data_radar_en_split |> 
  filter(str_count(body, "\\w+") > 5) |> 
  sample_n(10000)                         

Data_subset <- Data_subset |> 
  select(id, extraction_date, body, media_id)

# save subset
write_csv(Data_subset, "data/radar_subset_en.csv")
