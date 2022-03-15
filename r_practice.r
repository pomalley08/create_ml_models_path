library(tidyverse)
library(broom)

data_url <- 'https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/daily-bike-share.csv'
bike_data <- read_csv(data_url)

bike_data
glimpse(bike_data)

bike_data %>%
    ggplot(aes(x = temp, y = rentals)) +
    geom_point()

bike_data %>%
    ggplot(aes(x = temp)) +
    geom_histogram()

lm(rentals ~ . -dteday, data = bike_data) %>% summary()

