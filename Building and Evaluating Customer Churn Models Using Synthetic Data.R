

rm(list = ls(all.names = TRUE))

# Machine Learning Program in R
# End-to-end example with synthetic (dummy) data, model training, evaluation, and export
# Classification use case: predict customer churn (1 = churned, 0 = retained)

# -----------------------------
# 0) Setup
# -----------------------------
# If needed, uncomment the next lines to install packages
# install.packages(c("tidyverse","caret","pROC","randomForest"))

suppressPackageStartupMessages({
    library(tidyverse)
    library(caret)
    library(pROC)
    library(randomForest)
})

set.seed(1234)

# -----------------------------
# 1) Create realistic dummy data
# -----------------------------
n <- 3000

# Demographics and behavior
age <- round(rnorm(n, mean = 45, sd = 12))
age[age < 18] <- sample(18:22, sum(age < 18), replace = TRUE)
tenure_months <- pmax(1, round(rexp(n, rate = 1/24)))  # longer right tail
monthly_spend <- round(rlnorm(n, meanlog = log(70), sdlog = 0.35), 2)
customer_segment <- sample(c("Consumer","SMB","Enterprise"), n, replace = TRUE, prob = c(0.7,0.25,0.05))
is_promo <- rbinom(n, 1, 0.25)
num_support_tickets <- rpois(n, lambda = pmax(0.2, 1.2 - 0.02*pmin(tenure_months, 50)))
satisfaction <- pmin(10, pmax(1, round(7.5 - 0.015*tenure_months + rnorm(n, 0, 1.2) - 0.3*(num_support_tickets > 2))))
autopay <- rbinom(n, 1, 0.6)
contract_type <- sample(c("MonthToMonth","OneYear","TwoYear"), n, replace = TRUE, prob = c(0.6,0.25,0.15))
region <- sample(c("Northeast","South","Midwest","West"), n, replace = TRUE, prob = c(0.23,0.35,0.2,0.22))
add_on_features <- rpois(n, lambda = 1 + 0.02*tenure_months)

# Latent churn probability using a non-linear combination
linpred <- -2.0 +
    0.02*(age < 30) +
    0.8*(contract_type == "MonthToMonth") -
    0.015*tenure_months +
    0.01*(monthly_spend > 100) +
    0.4*(num_support_tickets >= 3) -
    0.12*satisfaction -
    0.4*autopay +
    0.15*(customer_segment == "Consumer") -
    0.10*(customer_segment == "Enterprise") +
    0.05*(region == "South") +
    0.02*add_on_features +
    0.25*is_promo

# Convert to probability through logistic function; add mild interaction noise
p_churn <- plogis(linpred + 0.15*(monthly_spend > 120 & satisfaction <= 6) - 0.1*(tenure_months > 36 & autopay == 1))

churn <- rbinom(n, 1, p_churn)

df <- tibble(
    churn = factor(churn, levels = c(0,1), labels = c("No","Yes")),
    age = age,
    tenure_months = tenure_months,
    monthly_spend = monthly_spend,
    customer_segment = factor(customer_segment),
    is_promo = factor(is_promo, levels = c(0,1), labels = c("No","Yes")),
    num_support_tickets = num_support_tickets,
    satisfaction = satisfaction,
    autopay = factor(autopay, levels = c(0,1), labels = c("No","Yes")),
    contract_type = factor(contract_type),
    region = factor(region),
    add_on_features = add_on_features
)

# -----------------------------
# 2) Train-test split
# -----------------------------
idx <- createDataPartition(df$churn, p = 0.75, list = FALSE)
train_df <- df[idx, ]
test_df  <- df[-idx, ]

# -----------------------------
# 3) Preprocessing pipeline
#    - One-hot encode factors
#    - Center and scale numeric predictors
# -----------------------------
x_cols <- setdiff(names(train_df), "churn")
pp <- preProcess(train_df[, x_cols], method = c("center","scale"))
train_x_num <- predict(pp, train_df[, x_cols])
test_x_num  <- predict(pp, test_df[, x_cols])

# One-hot encode for models that benefit from it
dmy <- dummyVars(~ ., data = train_x_num, fullRank = TRUE)
train_x <- as.data.frame(predict(dmy, newdata = train_x_num))
test_x  <- as.data.frame(predict(dmy, newdata = test_x_num))

train_y <- train_df$churn
test_y  <- test_df$churn

# -----------------------------
# 4) Cross-validation setup
# -----------------------------
cv_ctrl <- trainControl(
    method = "repeatedcv",
    number = 5,
    repeats = 2,
    classProbs = TRUE,
    summaryFunction = twoClassSummary,
    savePredictions = "final"
)

# -----------------------------
# 5) Train baseline model: Logistic Regression
# -----------------------------
glm_fit <- train(
    x = train_x,
    y = train_y,
    method = "glm",
    metric = "ROC",
    trControl = cv_ctrl,
    family = binomial()
)

# -----------------------------
# 6) Train non-linear model: Random Forest
# -----------------------------
rf_grid <- expand.grid(mtry = pmax(2, round(seq(2, ncol(train_x), length.out = 6))))
rf_fit <- train(
    x = train_x,
    y = train_y,
    method = "rf",
    metric = "ROC",
    trControl = cv_ctrl,
    tuneGrid = rf_grid,
    ntree = 500,
    importance = TRUE
)

# -----------------------------
# 7) Evaluate on the test set
# -----------------------------
# Predictions: probabilities and classes
glm_prob <- predict(glm_fit, test_x, type = "prob")[, "Yes"]
rf_prob  <- predict(rf_fit,  test_x, type = "prob")[, "Yes"]

glm_pred <- ifelse(glm_prob >= 0.5, "Yes", "No") %>% factor(levels = c("No","Yes"))
rf_pred  <- ifelse(rf_prob  >= 0.5, "Yes", "No") %>% factor(levels = c("No","Yes"))

# Confusion matrices
glm_cm <- confusionMatrix(glm_pred, test_y, positive = "Yes")
rf_cm  <- confusionMatrix(rf_pred,  test_y, positive = "Yes")

# ROC and AUC
glm_roc <- roc(response = test_y, predictor = glm_prob, levels = c("No","Yes"), direction = "<")
rf_roc  <- roc(response = test_y, predictor = rf_prob,  levels = c("No","Yes"), direction = "<")

cat("Test performance summary\n")
cat("--------------------------------------------------\n")
cat("Logistic Regression:\n")
print(glm_cm$byClass[c("Sensitivity","Specificity","Precision","Recall","F1")])
cat(sprintf("AUC: %.4f\n\n", as.numeric(auc(glm_roc))))

cat("Random Forest:\n")
print(rf_cm$byClass[c("Sensitivity","Specificity","Precision","Recall","F1")])
cat(sprintf("AUC: %.4f\n\n", as.numeric(auc(rf_roc))))

# -----------------------------
# 8) Compare models side by side
# -----------------------------
compare_tbl <- tibble(
    model = c("LogisticRegression","RandomForest"),
    Accuracy = c(glm_cm$overall["Accuracy"], rf_cm$overall["Accuracy"]),
    Kappa = c(glm_cm$overall["Kappa"], rf_cm$overall["Kappa"]),
    Sensitivity = c(glm_cm$byClass["Sensitivity"], rf_cm$byClass["Sensitivity"]),
    Specificity = c(glm_cm$byClass["Specificity"], rf_cm$byClass["Specificity"]),
    Precision = c(glm_cm$byClass["Precision"], rf_cm$byClass["Precision"]),
    Recall = c(glm_cm$byClass["Recall"], rf_cm$byClass["Recall"]),
    F1 = c(glm_cm$byClass["F1"], rf_cm$byClass["F1"]),
    AUC = c(as.numeric(auc(glm_roc)), as.numeric(auc(rf_roc)))
) %>% mutate(across(where(is.numeric), round, 4))

print(compare_tbl)

# -----------------------------
# 9) Variable importance and simple plots
# -----------------------------
# Variable importance for random forest
vi <- varImp(rf_fit, scale = TRUE)$importance %>%
    rownames_to_column("feature")

# When caret returns both "No" and "Yes", combine them
importance_cols <- setdiff(names(vi), "feature")

vi <- vi %>%
    mutate(Importance = rowMeans(across(all_of(importance_cols))))

rf_vi <- vi %>%
    arrange(desc(Importance)) %>%
    slice(1:20)

# Bar plot of top 20 important features
ggplot(rf_vi, aes(x = reorder(feature, Importance), y = Importance)) +
    geom_col() +
    coord_flip() +
    labs(title = "Random Forest Variable Importance (Top 20)",
         x = "Feature",
         y = "Importance")

# ROC curves overlay
roc_df <- bind_rows(
    tibble(tpr = glm_roc$sensitivities, fpr = 1 - glm_roc$specificities, model = "LogisticRegression"),
    tibble(tpr = rf_roc$sensitivities,  fpr = 1 - rf_roc$specificities,  model = "RandomForest")
)

ggplot(roc_df, aes(x = fpr, y = tpr, linetype = model)) +
    geom_line() +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    labs(title = "ROC Curves", x = "False Positive Rate", y = "True Positive Rate")

# -----------------------------
# 10) Export artifacts
# -----------------------------
# Predicted probabilities and classes
pred_out <- test_df %>%
    select(churn) %>%
    mutate(
        glm_prob = glm_prob,
        glm_pred = glm_pred,
        rf_prob = rf_prob,
        rf_pred = rf_pred
    )

# Write CSVs to working directory
write.csv(pred_out, "test_predictions.csv", row.names = FALSE)
write.csv(compare_tbl, "model_comparison.csv", row.names = FALSE)

# Save the best model (based on AUC here we choose RF as example)
saveRDS(rf_fit, file = "model_rf_fit.rds")

cat("Files saved: test_predictions.csv, model_comparison.csv, model_rf_fit.rds\n")

# -----------------------------
# 11) Inference example on new data
# -----------------------------
new_customers <- tibble(
    age = c(27, 58),
    tenure_months = c(3, 40),
    monthly_spend = c(130, 62),
    customer_segment = factor(c("Consumer","SMB"), levels = levels(df$customer_segment)),
    is_promo = factor(c("Yes","No"), levels = c("No","Yes")),
    num_support_tickets = c(4, 0),
    satisfaction = c(5, 9),
    autopay = factor(c("No","Yes"), levels = c("No","Yes")),
    contract_type = factor(c("MonthToMonth","TwoYear"), levels = levels(df$contract_type)),
    region = factor(c("South","Northeast"), levels = levels(df$region)),
    add_on_features = c(1, 3)
)

# Apply same preprocessing
new_x_num <- predict(pp, new_customers)
new_x <- as.data.frame(predict(dmy, newdata = new_x_num))

# Predict with saved RF model
rf_loaded <- readRDS("model_rf_fit.rds")
new_prob <- predict(rf_loaded, new_x, type = "prob")[, "Yes"]
new_pred <- ifelse(new_prob >= 0.5, "Yes", "No")

inference_out <- new_customers %>%
    mutate(pred_prob_churn = round(new_prob, 4), pred_class = new_pred)

print(inference_out)
