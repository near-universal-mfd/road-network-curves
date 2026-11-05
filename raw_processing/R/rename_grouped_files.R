# Locations start from ROAD_NETWORK_RAW (default: raw/, from the repository's root). Otherwise as the authors ran it. It needs the original UTD19 download, so it was not rerun for this package.
RAW_ROOT <- normalizePath(Sys.getenv("ROAD_NETWORK_RAW", "raw"), mustWork = FALSE)

library(dplyr)
library(readr)

# Source and Target directories
source_path <- file.path(RAW_ROOT, "cleaned")
target_path <- file.path(RAW_ROOT, "curves_as_reported")

# Which city each numbered file holds. Parts follow the city order in utd19_u.csv
# (2- accessing_data.R); 22 and 36 hold no data. `corre` is the city's spelling in the
# _corre files, `name` the network name used downstream (UTD19's spelling).
#
# The first version of this script paired the numbered files with the alphabetical list of
# _corre files by position. UTD19 sorts Konstanz as "constance" and Turin as "torino", while
# the _corre files say "konstanz" and "turin", so parts 9-17 and 33-35 each received the
# next city's name (fixed 31 Oct 2026). Naming from this table, and checking each file's
# dates against its city's _corre file, prevents that.
parts <- tribble(
  ~part, ~corre,       ~name,
   1,    "augsburg",   "augsburg",
   2,    "basel",      "basel",
   3,    "bern",       "bern",
   4,    "birmingham", "birmingham",
   5,    "bolton",     "bolton",
   6,    "bordeaux",   "bordeaux",
   7,    "bremen",     "bremen",
   8,    "cagliari",   "cagliari",
   9,    "konstanz",   "constance",
  10,    "darmstadt",  "darmstadt",
  11,    "essen",      "essen",
  12,    "frankfurt",  "frankfurt",
  13,    "graz",       "graz",
  14,    "groningen",  "groningen",
  15,    "hamburg",    "hamburg",
  16,    "innsbruck",  "innsbruck",
  17,    "kassel",     "kassel",
  18,    "london",     "london",
  19,    "losangeles", "losangeles",
  20,    "lucerne",    "luzern",
  21,    "madrid",     "madrid",
  23,    "manchester", "manchester",
  24,    "marseille",  "marseille",
  25,    "munich",     "munich",
  26,    "paris",      "paris",
  27,    "rotterdam",  "rotterdam",
  28,    "santander",  "santander",
  29,    "speyer",     "speyer",
  30,    "strasbourg", "strasbourg",
  31,    "stuttgart",  "stuttgart",
  32,    "taipei",     "taipeh",
  33,    "turin",      "torino",
  34,    "toronto",    "toronto",
  35,    "toulouse",   "toulouse",
  37,    "vilnius",    "vilnius",
  38,    "wolfsburg",  "wolfsburg",
  39,    "zurich",     "zurich"
)

days_in <- function(file) {
  read_csv(file, col_select = day, show_col_types = FALSE) |> pull(day) |> unique()
}

for (k in seq_len(nrow(parts))) {
  old_name <- file.path(target_path, paste0("data", parts$part[k], "_grouped.csv"))
  new_name <- file.path(target_path, paste0("data_", parts$name[k], "_grouped.csv"))
  corre    <- file.path(source_path, paste0("data_", parts$corre[k], "_corre.csv"))

  if (!file.exists(old_name)) {
    warning("File not found: ", old_name)
    next
  }
  # each city covers its own set of days (basel/bern and bremen/speyer/wolfsburg share
  # theirs), so a mismatch means the file holds another city
  if (!setequal(days_in(old_name), days_in(corre))) {
    stop(basename(old_name), " does not cover the days in ", basename(corre))
  }
  file.rename(old_name, new_name)
  message("Renamed: ", basename(old_name), "  →  ", basename(new_name))
}
