##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

#remove all temp files
rm(dl, ratings, movies, test_index, temp, movielens, removed)

#save copies of the databases to local folder
save(edx, validation, file = "MovieLens.RData")

#========================================================================
#questions and answers from quiz

#size of table / rows columns
dim(edx)

#how many 0 ratings
count(edx, edx$rating == 0)

#how many 3 ratings
count(edx, edx$rating == 3)
edx %>% filter(rating == 3) %>% tally()

#how many unique titles
edx %>% group_by(title) %>% summary(title)

length(unique(edx$movieId))
n_distinct(edx$movieId)

#how many unique users
n_distinct(edx$userId)

#how many unique genres
n_distinct(edx$genres)  

#How many of each of the following genres
sum(str_detect("Drama",edx$genre))
sum(str_detect("Comedy",edx$genre))
sum(str_detect("Thriller",edx$genre))
sum(str_detect("Romance",edx$genre))

#count of most rated movie
edx %>% group_by(movieId, title) %>%
  summarise( num = n()) %>%
  arrange(desc(num))

#count of most popular rating
edx %>% group_by(rating) %>%
  summarise( num = n()) %>%
  arrange(desc(num))

#are there more half than full ratings?
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()

#========================================================================
#check data

#take a look at structure of table
head(edx)

#add year from title as potential parameter
edx <- edx %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
validation <- validation %>% mutate(year = as.numeric(str_sub(title,-5,-2)))

#check for NAs
sum(is.na(edx$rating))
sum(is.na(edx$year))

#check distribution of ratings
edx %>%
  group_by(rating) %>%
  ggplot(aes(rating)) +
  geom_bar() +
  xlab("Rating") +
  ylab("No. Ratings") +
  ggtitle("Number of Ratings")

#check for 0 ratings
sum(edx$rating == 0)
#or check with summary
summary(edx)

#how many unique users and movie titles
edx %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

#check amount of ratings per user
edx %>%
  dplyr::count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30) +
  scale_x_log10() +
  xlab("No. Users") +
  ylab("No. Ratings") +
  ggtitle("Number of Ratings per User")

#check amount of ratings per movie
edx %>%
  dplyr::count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30) +
  scale_x_log10() +
  xlab("No. Movies") +
  ylab("No. Ratings") +
  ggtitle("Number of Ratings per Movie")

#check effect of time on ratings
edx %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth(method = "loess", span = 0.15, method.args = list(degree=1)) +
  ggtitle("Average ratings by time unit (week)")

#check effect of year of release on rating
edx %>% group_by(year) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(year, rating)) +
  geom_point() +
  geom_smooth(method = "loess", span = 0.5, method.args = list(degree=1)) +
  ggtitle("Average ratings by year of release")

edx %>%
  group_by(userId, movieId) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(movieId, userId, color = rating)) +
  geom_point()

#check amount of ratings per genre
edx %>%
  dplyr::count(genres) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30) +
  scale_x_log10() +
  xlab("No. Genres") +
  ylab("No. Ratings") +
  ggtitle("Number of Ratings per Genre")

#========================================================================
#function to calculate residual mean squared error
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
#========================================================================
#simple average model
#========================================================================

#take mean of all ratings 
mu <- mean(edx$rating)
mu

#check average against test set
naive_rmse <- RMSE(validation$rating, mu)
naive_rmse

#save first results to table
rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)

#========================================================================
#model for movie effects
#========================================================================

mu <- mean(edx$rating) 
#create table with movieID and the bias to be added to the mean
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

qplot(b_i, data = movie_avgs, bins = 10, color = I("black"))

#now make the predictions
predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
movie_model_rmse <- RMSE(predicted_ratings, validation$rating)

#save results to summary table
rmse_results <- rmse_results %>% add_row(tibble_row(method = "Movie effect model", RMSE = movie_model_rmse))

#========================================================================
#model for user effects
#========================================================================

edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

#calculate the bias to be added to the mean
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

qplot(b_u, data = user_avgs, bins = 10, color = I("black"))

#now make the predictions
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
user_model_rmse <- RMSE(predicted_ratings, validation$rating)

#save results to summary table
rmse_results <- rmse_results %>% add_row(tibble_row(method = "User effect model", RMSE = user_model_rmse))

#========================================================================
#model for effect of year of release
#========================================================================

edx %>% 
  group_by(year) %>% 
  summarize(b_y = mean(rating)) %>% 
  ggplot(aes(b_y)) + 
  geom_histogram(bins = 30, color = "black")

#calcualte the bias to be added to the mean
year_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(year) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u))

qplot(b_y, data = year_avgs, bins = 10, color = I("black"))

#now make the predictions
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  mutate(pred = mu + b_i + b_u + b_y) %>%
  pull(pred)
year_model_rmse <- RMSE(predicted_ratings, validation$rating)

#save results to summary table
rmse_results <- rmse_results %>% add_row(tibble_row(method = "Release year model", RMSE = year_model_rmse))
#========================================================================
#model for effect of genre
#========================================================================

edx %>% 
  group_by(genres) %>% 
  summarize(b_g = mean(rating)) %>% 
  ggplot(aes(b_g)) + 
  geom_histogram(bins = 30, color = "black")

#calculate the bias to be added to the mean
genre_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u - b_y))

qplot(b_g, data = genre_avgs, bins = 10, color = I("black"))

#now make the predictions
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_y + b_g) %>%
  pull(pred)
genre_model_rmse <- RMSE(predicted_ratings, validation$rating)

#save results to summary table
rmse_results <- rmse_results %>% add_row(tibble_row(method = "Genre model", RMSE = genre_model_rmse))

#========================================================================
#regularising
#========================================================================

#applying less significance to ratings with lower count
lambda <- 3
mu <- mean(edx$rating)
movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

tibble(original = movie_avgs$b_i, 
       regularlized = movie_reg_avgs$b_i, 
       n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

#now make the predictions
predicted_ratings <- validation %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)
ratings_w_model_rmse <- RMSE(predicted_ratings, validation$rating)

#save results to summary table
rmse_results <- rmse_results %>% add_row(tibble_row(method = "No. ratings weighted model", RMSE = ratings_w_model_rmse))

#========================================================================
#regularising - choosing a tuning parameter
#========================================================================

#create a sequence of lambdas to test for best fit
lambdas <- seq(0, 10, 0.5)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_y <- edx %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - b_i - b_u - mu)/(n()+l))
  
  b_g <- edx %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_y, by="year") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - b_y - mu)/(n()+l))
  
  predicted_ratings <- validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_y + b_g) %>%
    .$pred
  
  return(RMSE(validation$rating,predicted_ratings))
})
# Plot rmses vs lambdas to select the optimal lambda
qplot(lambdas, rmses)  
lambda <- lambdas[which.min(rmses)]
lambda

#save results to summary table
rmse_results <- rmse_results %>% add_row(tibble_row(method = "No. ratings opt. weighted model", RMSE = min(rmses)))
