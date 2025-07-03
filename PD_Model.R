# =================================================================
# Credit Risk Model - Final Project Script
# Author: Anusha
# Date: 2nd July 2025
# =================================================================

# setwd("~/Desktop/Case Study/Submission")

# -----------------------------------------------------------------
# 1. SETUP
# -----------------------------------------------------------------
# This section loads all the required R packages for the analysis.
# Each package provides a specific set of tools for our workflow.
# This section installs and loads all the libraries we will need.

library(tidyverse) # A collection of packages for data manipulation and visualization (e.g., ggplot2, dplyr).
library(pROC)      # A specialized package for calculating and visualizing ROC curves and AUC.
library(caret)     # Provides a comprehensive framework for training and evaluating machine learning models.
library(mice)      # An advanced library for imputing missing data using predictive models.
library(themis)    # A package that provides methods for handling class imbalance, including SMOTE.

# -----------------------------------------------------------------
# 2. DATA LOADING AND PREPARATION
# -----------------------------------------------------------------
# This section handles the entire data preparation pipeline. The goal is to
# create a clean, complete, and reliable dataset for modeling. All data
# cleaning is performed on the entire dataset before splitting to ensure consistency.

# --- Load the raw dataset from the CSV file ---
data <- read.csv("logreg_data.csv", sep = ";")

# 1. Count how many 'City' values are either NA or an empty string ""
missing_city_count <- sum(is.na(data$City) | data$City == "")
print(paste("Initial count of missing 'City' values (NA or empty string):", missing_city_count))

# 2. Count how many 'Loan.Amount' values are missing (NA)
missing_loan_amount_count <- sum(is.na(data$Loan.Amount))
print(paste("Initial count of missing 'Loan.Amount' values:", missing_loan_amount_count))

# 3. Count how many 'Year.of.birth' values are illogical (birth year is after application year)
illogical_yob_count <- sum(data$Year.of.birth > data$Year.of.application, na.rm = TRUE)
print(paste("Initial count of illogical 'Year.of.birth' values:", illogical_yob_count))

# --- Step 2.1: Handle Missing 'City' ---
# The 'City' column contains both NA values and empty strings (""). This code
# standardizes all missing city information into a new factor level, "Unknown",
# which preserves the records and treats the missingness as potentially predictive.

data <- data %>%
  mutate(City = as.character(City), # Ensure it's a character vector
         City = if_else(is.na(City) | City == "", "Unknown", City))


# --- Step 2.2: Handle Anomalous and Missing Numerical Data ---
# First, we identify and flag anomalous 'Year.of.birth' values (any year after
# the application year) by converting them to NA.

data <- data %>%
  mutate(Year.of.birth = if_else(Year.of.birth > Year.of.application, NA_integer_, Year.of.birth))

# Next, we use the MICE (Multivariate Imputation by Chained Equations) package
# to impute all remaining missing values (in 'Loan.Amount' and 'Year.of.birth').
# We use the 'pmm' (Predictive Mean Matching) method, which is a robust technique
# that fills missing values with plausible data from similar, real records.
set.seed(123) # Ensures reproducibility of the imputation.
imputed_data <- mice(data, m=1, method='pmm', seed=123, printFlag = FALSE)
data <- complete(imputed_data, 1)

# --- Final check to confirm the dataset is 100% clean ---
print("--- NA Count After Final Imputation ---")
sapply(data, function(x) sum(is.na(x)))


# -----------------------------------------------------------------
# 3. FEATURE ENGINEERING
# -----------------------------------------------------------------
# This section creates new, more insightful variables from the raw data
# and prepares the data types for modeling.

# Create the 'Age' variable, as age at time of application is a more direct
# and powerful predictor than year of birth.

data <- data %>%
  mutate(Age = Year.of.application - Year.of.birth)

# Convert all categorical variables to factors. This is a necessary step
# to ensure the modeling functions in R handle them correctly.

data <- data %>%
  mutate(
    Gender = as.factor(Gender),
    Education = as.factor(Education),
    City = as.factor(City), # Convert the cleaned City column to a factor
    DEFAULT = as.factor(DEFAULT)
  )

# Create a helper 'Age.Bracket' column for visualization purposes in the EDA.
data <- data %>%
  mutate(Age.Bracket = cut(Age,
                           breaks = seq(floor(min(Age)/10)*10, ceiling(max(Age)/10)*10, by = 10),
                           right = FALSE))

# -----------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------
# This section generates the key visualizations used to understand the
# data and to support the findings in the final presentation.

# --- Summary Statistics ---
summary(data)

data <- data %>%
  mutate(Age.Bracket = cut(Age,
                           breaks = seq(floor(min(Age)/10)*10, ceiling(max(Age)/10)*10, by = 10),
                           right = FALSE))


# --- Visualizations ---

# Plot 1: Proportion of Default by Age Group (Histogram)
# This visualization explores the relationship between customer age and default rate.
ggplot(data, aes(x = Age, fill = DEFAULT)) +
  geom_histogram(binwidth = 5, position = "fill", alpha = 0.7) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "Proportion of Default by Age Group",
       x = "Age",
       y = "Proportion of Default") +
  theme_minimal()

# Plot 2: Loan Amount by Default Status (Boxplot)
# This plot compares the distribution of loan amounts for defaulting vs. non-defaulting customers.
ggplot(data, aes(x = DEFAULT, y = Loan.Amount, fill = DEFAULT)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Loan Amount by Default Status",
       x = "Default Status (1 = Default)",
       y = "Loan Amount") +
  theme_minimal()

# Plot 3: Loan Amount vs. Age (Scatter Plot)
# This plot examines the relationship between two continuous variables to identify life-cycle trends.
ggplot(data, aes(x = Age, y = Loan.Amount)) +
  geom_point(alpha = 0.3, color = "blue") + # Makes points semi-transparent to see density
  geom_smooth() + # Adds a smooth trend line to see the average pattern
  labs(title = "Loan Amount vs. Age",
       x = "Applicant Age",
       y = "Loan Amount") +
  theme_minimal()

# Plot 4: Applicant vs. Loan Value Distribution by Age (Grouped Bar Chart)
# This chart provides a business-centric view by comparing the proportion of applicants
# to their corresponding share of the total loan value across different age brackets.
applicant_dist <- data %>%
  group_by(Age.Bracket) %>%
  summarise(count = n()) %>%
  mutate(prop = count / sum(count), type = "Percentage of Applicants")

loan_value_dist <- data %>%
  group_by(Age.Bracket) %>%
  summarise(Total.Loan.Amount = sum(Loan.Amount)) %>%
  mutate(prop = Total.Loan.Amount / sum(Total.Loan.Amount), type = "Percentage of Loan Value")

combined_dist <- bind_rows(applicant_dist, loan_value_dist)

ggplot(combined_dist, aes(x = Age.Bracket, y = prop, fill = type)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(title = "Applicant vs. Loan Value Distribution by Age",
       subtitle = "Comparing the share of applicants to their share of total loan value.",
       x = "Age Bracket",
       y = "Percentage",
       fill = "Metric") +
  theme_minimal() +
  theme(legend.position = "top")

# This section analyzes where the default risk is concentrated in terms
# of customer count vs. actual loan value.

# Define the threshold for "high-value" clients (top 15% by loan amount).
loan_threshold <- quantile(data$Loan.Amount, 0.85)

# Create a new column to categorize each customer.
data_with_value_segment <- data %>%
  mutate(Value.Segment = if_else(Loan.Amount >= loan_threshold, "High-Value", "Standard-Value"))

# Filter for only the customers who have defaulted.
defaulted_clients <- data_with_value_segment %>%
  filter(DEFAULT == 1)

# Plot 5: This chart shows what percentage of total defaulters are high-value.
defaulter_counts <- defaulted_clients %>%
  group_by(Value.Segment) %>%
  summarise(count = n()) %>%
  mutate(prop = count / sum(count) * 100)

ggplot(defaulter_counts, aes(x = "", y = prop, fill = Value.Segment)) +
  geom_bar(stat = "identity", width = 1, color = "white") +
  coord_polar("y", start = 0) +
  theme_void() +
  geom_text(aes(label = paste0(round(prop), "%")),
            position = position_stack(vjust = 0.5)) +
  labs(title = "Proportion of Defaulters by Customer Value",
       subtitle = "Comparing the number of high-value vs. standard-value defaulters.",
       fill = "Customer Segment")


# Plot 6: This chart shows what percentage of the total money lost to defaults comes from high-value
defaulted_value <- defaulted_clients %>%
  group_by(Value.Segment) %>%
  summarise(Total.Loan.Amount = sum(Loan.Amount)) %>%
  mutate(prop = Total.Loan.Amount / sum(Total.Loan.Amount) * 100)

ggplot(defaulted_value, aes(x = "", y = prop, fill = Value.Segment)) +
  geom_bar(stat = "identity", width = 1, color = "white") +
  coord_polar("y", start = 0) +
  theme_void() +
  geom_text(aes(label = paste0(round(prop), "%")),
            position = position_stack(vjust = 0.5)) +
  labs(title = "Proportion of Defaulted Loan Value by Customer Segment",
       subtitle = "Comparing the financial value of high-value vs. standard-value defaults.",
       fill = "Customer Segment")

# -----------------------------------------------------------------
# 5. MODEL BUILDING & EVALUATION
# -----------------------------------------------------------------
# This section details the process of building, validating, and evaluating
# the final predictive model using a robust, multi-step methodology.

# --- Step 5.1: Split Data into Training and Testing Sets ---
# We partition the clean data into an 80% training set (for building the model)
# and a 20% hold-out testing set (for unbiased evaluation).
set.seed(123)
train_index <- createDataPartition(data$DEFAULT, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

# --- Step 5.2: Automated Feature Selection ---
# To build a parsimonious and robust model, we first perform automated feature
# selection on the training data. We start with a full model and use backward
# stepwise selection to eliminate variables that are not statistically significant.
full_model <- glm(DEFAULT ~ . - Year.of.application - Year.of.birth - Age.Bracket,
                  data = train_data,
                  family = binomial(link = "logit"))

# We explicitly use the 'step' function from the 'stats' package to avoid
# namespace conflicts with other loaded libraries. This returns the optimal model formula.
final_model_formula <- formula(stats::step(full_model, direction = "backward", trace = 0))
print("--- Final Model Formula: ---")
print(final_model_formula)

# --- Step 5.3: Train Final Model with Advanced Validation ---
# We train our final model using 10-fold Cross-Validation for a robust
# performance estimate. We also apply SMOTE (Synthetic Minority Over-sampling
# Technique) during training to address the class imbalance in our data.
train_control <- trainControl(method = "cv", number = 10, sampling = "smote")
set.seed(123)
final_model_smote <- train(final_model_formula,
                           data = train_data,
                           method = "glm",
                           family = binomial(link = "logit"),
                           trControl = train_control)

# --- Step 5.4: Model Interpretation and Evaluation ---
# Print the cross-validation results and the final model summary.
print("--- K-Fold Cross-Validation Results (with SMOTE) ---")
print(final_model_smote)
print("--- Final Model Summary (trained on balanced data) ---")
summary(final_model_smote)

# Use the final model to make predictions on the unseen test set.
predictions <- predict(final_model_smote, newdata = test_data, type = "prob")
test_data$pred_prob <- predictions$'1'

# Classify the event based on a 0.5 probability threshold.
test_data$pred_class <- as.factor(ifelse(test_data$pred_prob > 0.5, 1, 0))

# --- Step 5.5: Calculate Final Performance Metrics ---
# Calculate standard industry performance metrics on the hold-out test set.
roc_curve <- roc(test_data$DEFAULT, test_data$pred_prob)
auc_value <- auc(roc_curve)
gini_coefficient <- 2 * auc_value - 1
print(paste("AUC on Hold-Out Test Set:", round(auc_value, 4)))
print(paste("Gini Coefficient on Hold-Out Test Set:", round(gini_coefficient, 4)))

calculate_ks <- function(predictions, actuals) {
  actuals_num <- as.numeric(as.character(actuals))
  pred_df <- data.frame(prob = predictions, default = actuals_num)
  pred_df <- pred_df[order(pred_df$prob, decreasing = TRUE), ]
  pred_df$cum_default <- cumsum(pred_df$default) / sum(pred_df$default)
  pred_df$cum_non_default <- cumsum(1 - pred_df$default) / sum(1 - pred_df$default)
  ks <- max(abs(pred_df$cum_default - pred_df$cum_non_default))
  return(ks)
}
ks_statistic <- calculate_ks(test_data$pred_prob, test_data$DEFAULT)
print(paste("K-S Statistic on Hold-Out Test Set:", round(ks_statistic, 4)))

# --- Step 5.6: Display Prediction Results in a Table ---
# Create a sample table to demonstrate the model's output on new data.
results_table <- test_data %>%
  select(Age, Education, Loan.Amount, # Key predictors
         DEFAULT,                     # The actual outcome
         pred_prob,                   # Our model's predicted probability
         pred_class)                  # Our model's final 0/1 prediction

# Round the probability for easier reading
results_table$pred_prob <- round(results_table$pred_prob, 4)

# Print the table
print("--- Prediction Summary Table (First 10 Test Customers) ---")
print(head(results_table, 10))

# -----------------------------------------------------------------
# 6. STRATEGIC SEGMENT ANALYSIS
# -----------------------------------------------------------------
# This section provides a deeper analysis of the model's performance, focusing
# on where it makes mistakes and how it performs on key business segments.

# --- Step 6.1: Prepare Data for Analysis (Enhanced Logic) ---
# Flag incorrect predictions to analyze error patterns.
test_data <- test_data %>%
  mutate(is_mistake = (DEFAULT != pred_class))

# Define the high-value threshold
loan_threshold <- quantile(data$Loan.Amount, 0.85)

# Categorize predictions with different rules for High-Value clients
test_data <- test_data %>%
  mutate(
    is_high_value = (Loan.Amount >= loan_threshold), # Helper column
    risk_category = case_when(
      # Stricter rules for High-Value clients
      is_high_value == TRUE & pred_prob < 0.30 ~ "Safe (HV)",
      is_high_value == TRUE & pred_prob <= 0.55 ~ "Grey Area (HV)",
      is_high_value == TRUE & pred_prob > 0.55 ~ "Risky (HV)",
      
      # Original rules for Standard-Value clients
      is_high_value == FALSE & pred_prob < 0.40 ~ "Safe",
      is_high_value == FALSE & pred_prob <= 0.60 ~ "Grey Area",
      is_high_value == FALSE & pred_prob > 0.60 ~ "Risky"
    )
  )

# --- Step 6.2: Overall Mistake Analysis ---
cat("\n--- Confusion Matrix (All Customers) ---\n")
print(confusionMatrix(test_data$pred_class, test_data$DEFAULT, positive="1"))


# --- Step 6.3: Visualize Mistake Distribution (SEPARATED)---

# First, split the test data into Standard and High-Value segments
standard_value_test_data <- test_data %>% filter(is_high_value == FALSE)
high_value_test_data <- test_data %>% filter(is_high_value == TRUE)

# New Plot 6a: Mistake Analysis for Standard-Value Clients
ggplot(standard_value_test_data, aes(x = is_mistake, fill = risk_category)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "Where Do Mistakes Occur? (Standard-Value Clients)",
       x = "Prediction Outcome",
       y = "Percentage",
       fill = "Risk Category") +
  scale_x_discrete(labels = c("Correct Prediction", "Incorrect Prediction")) +
  theme_minimal()

# New Plot 6b: Mistake Analysis for High-Value Clients
if (nrow(high_value_test_data) > 1) {
  ggplot(high_value_test_data, aes(x = is_mistake, fill = risk_category)) +
    geom_bar(position = "fill") +
    scale_y_continuous(labels = scales::percent_format()) +
    labs(title = "Where Do Mistakes Occur? (High-Value Clients)",
         subtitle = paste("Analysis of customers with Loan Amount >=", round(loan_threshold)),
         x = "Prediction Outcome",
         y = "Percentage",
         fill = "Risk Category") +
    scale_x_discrete(labels = c("Correct Prediction", "Incorrect Prediction")) +
    theme_minimal()
}




