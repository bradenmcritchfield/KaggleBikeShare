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
biketest <- vroom("./test.csv")

##Cleaning Step
  ##Recatergorize weather "4" value with "3" since there is only one occurrence
  biketrain <- biketrain %>%
    select(1:9, 12) %>%
    mutate(count = log(count)) #transform count to log scale


##Engineering Step
    #other variables
    #would aggregating day windspeed/humidity/temperature help?
    #night/day
    
  library(tidymodels)
    my_recipe <- recipe(count ~ ., biketrain)    %>%
    #  step_date(datetime, features = "dow") %>% #get day of week
      step_time(datetime, features = "hour") %>% #get hour
      step_zv(all_predictors()) %>% #remove any predictors with no variance
      step_rm(atemp) %>%
       #create interaction between season and weather
      step_mutate(weather = ifelse(weather == 4, 3, weather), weather = as.factor(weather), season = as.factor(season)) %>%
      step_interact(terms = ~ temp:daetime_hour) # %>% #turn weather and season into factors
    prepped_recipe <- prep(my_recipe)      
    DataExplorer::plot_correlation(bake(prepped_recipe, new_data = biketrain))
    

#################    
#Linear Analysis
#################
    my_mod <- linear_reg() %>% #Type of model
      set_engine("lm") # Engine = What R function to use
  bike_workflow <- workflow() %>%
    add_recipe(my_recipe) %>%
    add_model(my_mod) %>%
    fit(data = biketrain) #Fit workflow
  
  extract_fit_engine(bike_workflow) %>% summary()
  
  bikepredictions <- predict(bike_workflow, new_data = biketest)

  submission <- bikepredictions %>%
    mutate(.pred = ifelse(.pred < 0, 0, .pred)) %>%
    mutate(datetime = biketest$datetime) %>%
    mutate(datetime=as.character(format(datetime)))  %>%
    mutate(count = .pred) %>%
    select(2, 3)

  vroom_write(submission, "submission.csv", delim = ",")    
  
  
  
############################
  #Poisson Regression Model
############################
  
library(poissonreg)
pois_mod <- poisson_reg() %>% #Type of model
    set_engine("glm") # GLM = generalized linear model 45
bike_pois_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(pois_mod) %>%
fit(data = biketrain) # Fit the workflow

extract_fit_engine(bike_pois_workflow) %>% summary()

bike_predictions <- predict(bike_pois_workflow,
new_data=biketest) # Use fit to predict

submission <- bike_predictions %>%
  #mutate(.pred = exp(.pred)) %>%
  mutate(datetime = biketest$datetime) %>%
  mutate(datetime=as.character(format(datetime)))  %>%
  mutate(count = exp(.pred)) %>% #transform back to original scale
  select(2, 3)

vroom_write(submission, "submissionpoisson.csv", delim = ",")   


###############################
  #Penalized Regression
###############################


library(tidymodels)
library(poissonreg)
my_recipe <- recipe(count ~ ., biketrain)    %>%
  #  step_date(datetime, features = "dow") %>% #get day of week
  step_time(datetime, features = "hour") %>% #get hour
  step_mutate(weather = ifelse(weather == 4, 3, weather), weather = as.factor(weather), season = as.factor(season)) %>% #turn weather and season into factors
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())%>%
  step_zv(all_predictors())#remove any predictors with no variance
prepped_recipe <- prep(my_recipe)      
bake(prepped_recipe, new_data = biketrain)
DataExplorer::plot_correlation(bake(prepped_recipe, new_data = biketrain))

preg_model <- linear_reg(penalty=.25, mixture=0) %>% #Set model and tuning
set_engine("glmnet") # Function to fit in R
preg_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(preg_model) %>%
fit(data=biketrain)
bike_predictions_pen <- predict(preg_wf, new_data=biketest)

submission <- bike_predictions_pen %>%
  mutate(datetime = biketest$datetime) %>%
  mutate(datetime=as.character(format(datetime)))  %>%
  mutate(count = exp(.pred)) %>% #transform back to original scale
  select(2, 3)

vroom_write(submission, "submissionpenalized.csv", delim = ",")   

###################
#Random Forest
######################
library(tidymodels)
library(poissonreg)
my_recipe <- recipe(count ~ ., biketrain)    %>%
  #  step_date(datetime, features = "dow") %>% #get day of week
  step_time(datetime, features = "hour") %>% #get hour
  step_rm(datetime)%>%
  step_rm(temp) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather), weather = as.factor(weather), season = as.factor(season)) %>% #turn weather and season into factors
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())%>%
  step_zv(all_predictors())#remove any predictors with no variance
prepped_recipe <- prep(my_recipe)      
bake(prepped_recipe, new_data = biketrain)
DataExplorer::plot_correlation(bake(prepped_recipe, new_data = biketrain))

rf_model <- rand_forest(mode="regression") %>% #Set model and tuning
  set_engine("ranger") # Function to fit in R
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_model) %>%
  fit(data=biketrain)
bike_predictions_pen <- predict(preg_wf, new_data=biketest)

submission <- bike_predictions_pen %>%
  mutate(datetime = biketest$datetime) %>%
  mutate(datetime=as.character(format(datetime)))  %>%
  mutate(count = exp(.pred)) %>% #transform back to original scale
  select(2, 3)

vroom_write(submission, "submissionrf.csv", delim = ",")   



