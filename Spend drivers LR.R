# Robust, no-pipe R script for logistic regression:
# Predicting HighSpend (above-median Total_Amount) from behaviour metrics.
# Usage: set file_path to your CSV file (default assumes file is in working directory).

# ---------------------------
# 0. Settings: change this if needed
# ---------------------------
file_path <- "C:/Users/aishu/Downloads/ecom behaviour/ecommerce_customer_behavior_dataset_v2.csv"  # <- change to full path if needed

# ---------------------------
# 1. Helper: install & load packages safely
# ---------------------------
safe_load <- function(pkgs){
  for(pkg in pkgs){
    if(! (pkg %in% rownames(installed.packages())) ){
      install.packages(pkg, repos = "https://cloud.r-project.org")
    }
    suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  }
}
safe_load(c("broom", "ggplot2", "pROC", "gridExtra"))

# ---------------------------
# 2. Read data
# ---------------------------
if(!file.exists(file_path)){
  stop(paste0("File not found: ", file_path, "\nSet 'file_path' to the correct CSV location."))
}
df <- read.csv(file_path, stringsAsFactors = FALSE)

# If Date exists and you want to parse: (optional)
if("Date" %in% names(df)){
  parsed <- try(as.Date(df$Date), silent = TRUE)
  if(!inherits(parsed, "try-error")){
    df$Date <- parsed
  }
}

# ---------------------------
# 3. Build binary target: HighSpend (above-median Total_Amount)
# ---------------------------
if(!("Total_Amount" %in% names(df))){
  stop("Column 'Total_Amount' not found in data.")
}
median_spend <- median(df$Total_Amount, na.rm = TRUE)
df$HighSpend <- ifelse(is.na(df$Total_Amount), NA, as.integer(df$Total_Amount > median_spend))

# Remove rows with NA in target or predictors we'll use
required_cols <- c("HighSpend", "Pages_Viewed", "Session_Duration_Minutes", "Discount_Amount", "Quantity")
missing_cols <- setdiff(required_cols, names(df))
if(length(missing_cols) > 0){
  stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
}
complete_idx <- complete.cases(df[, required_cols])
df_clean <- df[complete_idx, ]

# ---------------------------
# 4. Fit logistic regression (GLM)
# ---------------------------
for(col in c("Pages_Viewed","Session_Duration_Minutes","Discount_Amount","Quantity")){
  df_clean[[col]] <- as.numeric(df_clean[[col]])
}
glm_model <- glm(HighSpend ~ Pages_Viewed + Session_Duration_Minutes + Discount_Amount + Quantity,
                 data = df_clean, family = binomial(link = "logit"))

# ---------------------------
# 5. Model summary & odds ratios
# ---------------------------
cat("----- GLM Summary -----\n")
print(summary(glm_model))

cat("\n----- Odds Ratios (exp(coef)) with 95% CI -----\n")
coefs <- coef(glm_model)
ci <- confint(glm_model)
or_table <- data.frame(
  Term = names(coefs),
  Coefficient = as.numeric(coefs),
  OR = exp(as.numeric(coefs)),
  CI_lower = exp(ci[,1]),
  CI_upper = exp(ci[,2]),
  row.names = NULL,
  stringsAsFactors = FALSE
)
print(or_table)

# ---------------------------
# 6. Predicted probabilities & threshold metrics
# ---------------------------
df_clean$pred_prob <- predict(glm_model, type = "response")
df_clean$pred_label <- ifelse(df_clean$pred_prob >= 0.5, 1, 0)

conf_mat <- table(Actual = df_clean$HighSpend, Predicted = df_clean$pred_label)
cat("\n----- Confusion Matrix (threshold=0.5) -----\n")
print(conf_mat)
accuracy <- sum(diag(conf_mat)) / sum(conf_mat)
cat(sprintf("Accuracy: %.3f\n", accuracy))

roc_obj <- pROC::roc(df_clean$HighSpend, df_clean$pred_prob, quiet = TRUE)
auc_val <- pROC::auc(roc_obj)
cat(sprintf("AUC: %.4f\n", auc_val))

# ---------------------------
# 7. Visualizations (ggplot2) — create plots
# ---------------------------
tidy_df <- broom::tidy(glm_model)
tidy_df$term <- factor(tidy_df$term, levels = tidy_df$term[order(tidy_df$estimate)])
p1 <- ggplot(tidy_df, aes(x = term, y = estimate)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Logistic Regression Coefficients", x = "Term", y = "Coefficient (log-odds)") +
  theme_minimal()

p2 <- ggplot(df_clean, aes(x = Pages_Viewed, y = pred_prob)) +
  geom_jitter(alpha = 0.3, width = 0.2, height = 0) +
  geom_smooth(method = "loess", se = TRUE) +
  labs(title = "Predicted Probability of High Spend vs Pages Viewed",
       x = "Pages Viewed", y = "Predicted Probability") +
  theme_minimal()

p3 <- ggplot(df_clean, aes(x = Session_Duration_Minutes, y = pred_prob)) +
  geom_jitter(alpha = 0.3, width = 0, height = 0) +
  geom_smooth(method = "loess", se = TRUE) +
  labs(title = "Predicted Probability of High Spend vs Session Duration (min)",
       x = "Session Duration (min)", y = "Predicted Probability") +
  theme_minimal()

roc_df <- data.frame(
  tpr = rev(roc_obj$sensitivities),
  fpr = rev(1 - roc_obj$specificities)
)
p4 <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(color = "darkred") +
  geom_abline(linetype = "dashed") +
  labs(title = paste0("ROC Curve (AUC = ", round(as.numeric(auc_val), 4), ")"),
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal()

# ---------------------------
# 7b. Display plots in the plot pane (explicit prints)
# ---------------------------
# For interactive RStudio/REPL: grid.arrange will show.
# For non-interactive contexts (sourced scripts), explicitly print each ggplot.
# We do both: attempt grid.arrange (nice layout), but also print individually to ensure display.
try({
  # attempt a 2x2 arrange in the plot area
  gridExtra::grid.arrange(p1, p2, p3, p4, ncol = 2)
}, silent = TRUE)

# Ensure each plot is printed to the active device (works everywhere)
print(p1)
print(p2)
print(p3)
print(p4)

# ---------------------------
# 7c. Save plots to PNG files (kept as before)
# ---------------------------
ggsave("logistic_coefficients.png", plot = p1, width = 7, height = 4)
ggsave("pred_vs_pages.png", plot = p2, width = 7, height = 4)
ggsave("pred_vs_session.png", plot = p3, width = 7, height = 4)
ggsave("roc_curve.png", plot = p4, width = 7, height = 4)

# ---------------------------
# 8. Save model outputs
# ---------------------------
write.csv(or_table, "logistic_odds_ratios.csv", row.names = FALSE)
write.csv(df_clean[, c("Pages_Viewed","Session_Duration_Minutes","Discount_Amount","Quantity","HighSpend","pred_prob","pred_label")],
          "logistic_predictions.csv", row.names = FALSE)

cat("\nAll done. Plots printed to the plot pane and saved to PNG files; results saved as CSVs.\n")
