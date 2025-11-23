# Corrected Random Forest classification script
file_path <- "C:/Users/aishu/Downloads/ecom behaviour/ecommerce_customer_behavior_dataset_v2.csv"

# --- load packages
pkgs <- c("randomForest","pROC","ggplot2","caret")
for(p in pkgs){
  if(!(p %in% rownames(installed.packages()))) install.packages(p, repos = "https://cloud.r-project.org")
  suppressPackageStartupMessages(library(p, character.only = TRUE))
}

# --- read data
if(!file.exists(file_path)) stop("File not found: set file_path correctly")
df <- read.csv(file_path, stringsAsFactors = FALSE)

# --- create target (factor!) and predictors
if(!("Total_Amount" %in% names(df))) stop("Total_Amount column missing")
median_spend <- median(df$Total_Amount, na.rm = TRUE)

# IMPORTANT: create factor target (character -> factor), not numeric 0/1
df$HighSpend <- ifelse(is.na(df$Total_Amount), NA, ifelse(df$Total_Amount > median_spend, "High", "Low"))
df$HighSpend <- factor(df$HighSpend, levels = c("Low","High"))  # order: Low = negative class, High = positive

# predictors
features <- c("Pages_Viewed","Session_Duration_Minutes","Discount_Amount","Quantity")
missing_cols <- setdiff(c(features, "HighSpend"), names(df))
if(length(missing_cols)>0) stop(paste("Missing columns:", paste(missing_cols, collapse = ", ")))

# coerce predictors to numeric and drop missing rows
for(col in features) df[[col]] <- as.numeric(as.character(df[[col]]))
df2 <- df[complete.cases(df[, c(features, "HighSpend")]), ]

# --- quick check: class of target
cat("Class of HighSpend: ", class(df2$HighSpend), "\n")
cat("Levels: ", paste(levels(df2$HighSpend), collapse = ", "), "\n")
print(table(df2$HighSpend))

# --- stratified train/test split
set.seed(42)
train_idx <- caret::createDataPartition(df2$HighSpend, p = 0.75, list = FALSE)
train <- df2[train_idx, ]
test  <- df2[-train_idx, ]

# --- fit Random Forest classifier
set.seed(42)
rf_model <- randomForest(HighSpend ~ Pages_Viewed + Session_Duration_Minutes + Discount_Amount + Quantity,
                         data = train, ntree = 500, importance = TRUE)

# Confirm model type
cat("RandomForest model type (classification if 'type' shows 'classification'):\n")
print(rf_model)   # will show type: classification/regression

# --- predictions: probabilities & classes (classification)
pred_prob <- predict(rf_model, test, type = "prob")[, "High"]  # probability of the "High" class
pred_class <- predict(rf_model, test, type = "response")

# --- confusion matrix & metrics
conf_mat <- table(Actual = test$HighSpend, Predicted = pred_class)
print(conf_mat)
accuracy <- sum(diag(conf_mat)) / sum(conf_mat)
cat(sprintf("Accuracy: %.4f\n", accuracy))

# AUC / ROC
roc_obj <- pROC::roc(as.numeric(test$HighSpend == "High"), pred_prob) # numeric TRUE/FALSE
auc_val <- pROC::auc(roc_obj)
cat(sprintf("AUC: %.4f\n", auc_val))

# --- feature importance
imp <- as.data.frame(importance(rf_model))
imp$feature <- rownames(imp)
imp <- imp[order(-imp$MeanDecreaseGini), , drop = FALSE]
print(imp)

# --- plots
library(ggplot2)
p_imp <- ggplot(imp, aes(x = reorder(feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_col(fill = "steelblue") + coord_flip() +
  labs(title = "Random Forest Feature Importance", x = "", y = "MeanDecreaseGini") +
  theme_minimal()
print(p_imp)

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  

# --- save outputs
write.csv(imp, "rf_feature_importance.csv", row.names = FALSE)
write.csv(data.frame(test[, features], y_true = test$HighSpend, y_prob = pred_prob, y_pred = pred_class),
          "rf_test_predictions.csv", row.names = FALSE)

cat("Random Forest classification completed. Plots printed and artifacts saved.\n")
