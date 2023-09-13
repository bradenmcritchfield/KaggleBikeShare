#Data Cleaning, Engineering, and Wrangling
#ENgineering

#Create a new R File BikeShareAnalysis.R
#Well-documented cleaning section, perform at least 1 cleaning step (only 1 day with weather = 4)
# perform at least 2 feature engineering
#Take a screenshot of about 10 rows of your new clean, engineered data set and share to LS
#add/commit/push file to GitHub

library(tidyverse)
library(vroom)

biketrain <- vroom("./train.csv")

##Cleaning Step
  ##Recatergorize weather "4" value with "3" since there is only one occurence
  biketrain1 <- biketrain %>%
    mutate(weather = ifelse(weather == 4, 3, weather))


##Engineering Step
  library(tidymodels)
    my_recipe <- recipe(count ~ ., biketrain1)    %>%
      step_date(datetime, features = "dow") %>% #get day of week
      step_time(datetime, features = "hour") %>% #get hour
      step_rm(casual)%>% #remove casual column
      step_rm(registered) %>% #remove registered column
      step_zv(all_predictors()) %>% #remove any predictors with no variance
      step_mutate(weather = as.factor(weather), season = as.factor(season))#turn weather and season into factors
    prepped_recipe <- prep(my_recipe)      
    bake(prepped_recipe, new_data = biketrain1)

