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


################################################
#Tuning Models
###############################################
library(tidymodels)
library(poissonreg)


#Penalized regression model
preg_model <- linear_reg(penalty=tune(), mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

#Set workflow
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model)

##Grid of values to tune over
tuning_grid <- grid_regular(penalty(), mixture(), levels = 10)

## Split data for CV
folds <- vfold_cv(biketrain, v = 5, repeats = 1)

## Run the CV
CV_results <- preg_wf %>%
  tune_grid(resamples = folds, grid = tuning_grid, metrics=metric_set(rmse, mae, rsq))

## Plot Results 
collect_metrics(CV_results) %>% # Gathers metrics into DF
filter(.metric=="rmse") %>%
ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
geom_line()
## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("rmse")

preg_model <- linear_reg(penalty=as.numeric(bestTune[1,1]), mixture=as.numeric(bestTune[1,1])) %>% #Set model and tuning
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


########################################################################
#Regression Trees
########################################################################

library(tidymodels)
my_mod <- decision_tree(tree_depth = tune(),
            cost_complexity = tune(),
            min_n=tune()) %>% #Type of model
  set_engine("rpart") %>% # Engine = What R function to use
  set_mode("regression")

## Create a workflow with model & recipe
prt_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

## Set up grid of tuning values
tuning_grid <- grid_regular(tree_depth(), cost_complexity(), min_n(), levels = 5)
## Set up K-fold CV
folds <- vfold_cv(biketrain, v = 5, repeats = 1)
## Find best tuning parameters
CV_results <- prt_wf %>%
  tune_grid(resamples = folds, grid = tuning_grid, metrics=metric_set(rmse, mae, rsq))
bestTune <- CV_results %>%
  select_best("rmse")
## Finalize workflow and predict

my_mod <- decision_tree(tree_depth = 15,
                        cost_complexity = 0,
                        min_n=30) %>% #Type of model
  set_engine("rpart") %>% # Engine = What R function to use
  set_mode("regression")
prt_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data=biketrain)
bike_predictions_rt <- predict(prt_wf, new_data=biketest)

submission <- bike_predictions_rt %>%
  mutate(datetime = biketest$datetime) %>%
  mutate(datetime=as.character(format(datetime)))  %>%
  mutate(count = exp(.pred)) %>% #transform back to original scale
  select(2, 3)

vroom_write(submission, "submissionregtree.csv", delim = ",")  


#################################################################################
# Random Forests
#################################################################################
install.packages("ranger")
library(tidymodels)
my_mod <- rand_forest(mtry = tune(),
                        min_n=tune(),
                        trees=500) %>% #Type of model
        set_engine("ranger") %>% # What R function to use
        set_mode("regression") 
## Create a workflow with model & recipe
randfor_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)
## Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range = c(1,(ncol(biketrain)-1))), min_n(), levels = 5)
## Set up K-fold CV
folds <- vfold_cv(biketrain, v = 5, repeats = 1)
## Find best tuning parameters
CV_results <- randfor_wf %>%
  tune_grid(resamples = folds, grid = tuning_grid, metrics=metric_set(rmse, mae, rsq))
bestTune <- CV_results %>%
  select_best("rmse")
## Finalize workflow and predict
my_mod_official <- rand_forest(mtry = 9,
                               min_n=2,
                               trees=500) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("regression")
randfor_wf_official <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod_official) %>%
  fit(data=biketrain)
bike_predictions_rf <- predict(randfor_wf_official, new_data=biketest)

submission <- bike_predictions_rf %>%
  mutate(datetime = biketest$datetime) %>%
  mutate(datetime=as.character(format(datetime)))  %>%
  mutate(count = exp(.pred)) %>% #transform back to original scale
  select(2, 3)

vroom_write(submission, "submissionrandforest.csv", delim = ",")  

###################################################################################################
#Stacking
###################################################################################################
library(stacks)
library(tidyverse)
library(tidymodels)
library(vroom)

biketrain <- vroom("./train.csv")
biketest <- vroom("./test.csv")

##Cleaning Step
biketrain <- biketrain %>%
  select(1:9, 12) %>%
  mutate(count = log(count)) #transform count to log scale

#make recipe
my_recipe <- recipe(count ~ ., biketrain)    %>%
  #  step_date(datetime, features = "dow") %>% #get day of week
  step_time(datetime, features = "hour") %>% #get hour
  step_date(datetime, features = "year") %>% #get year
  step_rm(datetime)%>%
  step_rm(temp) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather), weather = as.factor(weather), season = as.factor(season)) %>% #turn weather and season into factors
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())%>%
  step_zv(all_predictors())#remove any predictors with no variance
#prepped_recipe <- prep(my_recipe)      
#bake(prepped_recipe, new_data = biketrain)

##Split folds
folds <- vfold_cv(biketrain, v = 5, repeats = 1)

## Create a control grid
untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples()

##Penalized regression model
preg_model <- linear_reg(penalty=tune(), mixture = tune()) %>%
  set_engine("glmnet")

  #Set workflow
    preg_wf <- workflow() %>%
      add_recipe(my_recipe) %>%
      add_model(preg_model)
  #grid of values to tune over
    preg_tuning_grid <- grid_regular(penalty(), mixture(), levels = 5)
  #Run the CV
    preg_models <- preg_wf %>%
      tune_grid(resamples = folds, grid = preg_tuning_grid, metrics = metric_set(rmse, mae, rsq), control = untunedModel)
    
##Libear regression model
    #library(poissonreg)
    lin_reg <- linear_reg() %>% #Type of model
      set_engine("lm") # GLM = generalized linear model 45
    lin_reg_wf <- workflow() %>%
      add_recipe(my_recipe) %>%
      add_model(lin_reg)
    lin_reg_model <- fit_resamples(
      lin_reg_wf, resamples = folds, control = tunedModel)
    
##Random Forest model
    rand_for_mod <- rand_forest(mtry = tune(),
                          min_n=tune(),
                          trees=500) %>% #Type of model
      set_engine("ranger") %>% # What R function to use
      set_mode("regression") 
    randfor_wf <- workflow() %>%
      add_recipe(my_recipe) %>%
      add_model(rand_for_mod)
    rand_for_tg <- grid_regular(mtry(range = c(1,(ncol(biketrain)-1))), min_n(), levels = 5)
    rand_for_models <- randfor_wf %>% tune_grid(resamples = folds, grid = rand_for_tg, metrics = metric_set(rmse, mae, rsq), control = untunedModel)
  
##Specify models to include
    my_stack  <- stacks() %>%
      add_candidates(lin_reg_model) %>%
      add_candidates(rand_for_models)
    
##Fit the stacked model
    stack_mod <- my_stack %>%
      blend_predictions() %>%
      fit_members()
    
bike_predictions_stacking <- stack_mod %>% 
      predict(new_data =biketest)

submission <- bike_predictions_stacking %>%
  mutate(datetime = biketest$datetime) %>%
  mutate(datetime=as.character(format(datetime)))  %>%
  mutate(count = exp(.pred)) %>% #transform back to original scale
  select(2, 3)

vroom_write(submission, "submissionstacking.csv", delim = ",") 


##################################################################
# BART
##################################################################
my_BART_mod <- bart(mode ="regression",
                    engine = "dbarts", trees =20)

BART_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_BART_mod) %>%
  fit(data=biketrain)
bike_predictions_rf <- predict(BART_wf, new_data=biketest)

submission <- bike_predictions_rf %>%
  mutate(datetime = biketest$datetime) %>%
  mutate(datetime=as.character(format(datetime)))  %>%
  mutate(count = exp(.pred)) %>% #transform back to original scale
  select(2, 3)

vroom_write(submission, "submissionBART.csv", delim = ",")  
