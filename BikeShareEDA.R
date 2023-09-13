##

##Bike Share EDA Code
##

##Libraries

library(tidyverse)
library(vroom)

bike <- vroom("./train.csv")

##Perform EDA and identify key features

dplyr::glimpse(bike)
skimr::skim(bike)






library(patchwork)
plot1 <- DataExplorer::plot_correlation(bike)
plot2 <- DataExplorer::plot_histogram(bike)
plot3 <- DataExplorer::plot_intro(bike)
plot4 <- ggplot(data=bike, mapping=aes(x = datetime, y = count)) +
  geom_point() +
  labs(title="Relationship between datetime and count")

(plot1 + plot2)/(plot3+plot4)


##Create 4 panel EDA that shows 4 different key features of the dataset
## Save and upload a png file of the 4 panel plot to LS (use export in the image panel)
##Git add/commit/push your EDA to GitHub

