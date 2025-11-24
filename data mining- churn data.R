library(readr)
library(dplyr)
library(randomForest)
library(caret)

# Load the dataset
library(readr)
ecommerce_customer_behavior_dataset_v2 <- read_csv("dataset/ecommerce_customer_behavior_dataset_v2.csv")
View(ecommerce_customer_behavior_dataset_v2)

# Data preparation:
# Assume 'IsReturningCustomer' is the churn indicator (TRUE = returning, FALSE = churned)
# Convert to factor
data <- ecommerce_customer_behavior_dataset_v2 %>%
  mutate(Is_Returning_Customer = as.factor(Is_Returning_Customer))

# Select relevant behavioural variables and remove IDs and leakage columns
feature_vars <- c("Age", "Gender", "City", "Product_Category", "Unit_Price", "Quantity", "Discount_Amount",
                  "Total_Amount", "Payment_Method", "Device_Type", "Session_Duration_Minutes", "Pages_Viewed",
                  "Delivery_Time_Days", "Customer_Rating")
model_data <- data %>% select(all_of(c(feature_vars, "Is_Returning_Customer"))) %>%
  na.omit() # Remove missing values

# Split data into train/test
set.seed(123)
train_index <- createDataPartition(model_data$Is_Returning_Customer, p = 0.7, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

# Random Forest Model
rf_model <- randomForest(Is_Returning_Customer ~ ., data = train_data, ntree = 100, importance = TRUE)

# Predictions
rf_pred <- predict(rf_model, newdata = test_data)

# Confusion Matrix
conf_matrix <- confusionMatrix(rf_pred, test_data$Is_Returning_Customer)
print(conf_matrix)

# Variable Importance
importance_df <- data.frame(Feature = rownames(rf_model$importance),
                            Importance = rf_model$importance[, "MeanDecreaseGini"])
importance_df <- importance_df %>% arrange(desc(Importance))
print(importance_df)

#accuracy of model
accuracy <- sum(rf_pred == test_data$Is_Returning_Customer) / nrow(test_data)
print(paste("Model Accuracy:", round(accuracy * 100, 2), "%"))
#ROC CURVE and AUC
library(pROC)
rf_prob <- predict(rf_model, newdata = test_data, type = "prob")[,2]
roc_curve <- roc(test_data$Is_Returning_Customer, rf_prob)
auc_value <- auc(roc_curve)
print(paste("AUC:", round(auc_value, 4)))
plot(roc_curve, main = "ROC Curve for Random Forest Model")
# Save the model
saveRDS(rf_model, "random_forest_ecommerce_model.rds")
# Load the model (example)
# loaded_model <- readRDS("random_forest_ecommerce_model.rds")
# Make predictions with the loaded model
 
# loaded_pred <- predict(loaded_model, newdata = test_data)
# print(confusionMatrix(loaded_pred, test_data$Is_Returning_Customer))


