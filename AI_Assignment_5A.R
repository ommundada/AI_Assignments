# ============================================================
# 0. LOAD & CLEAN DATA
# ============================================================

df <- `2020_bn_nb_data`        # your imported dataset
df <- as.data.frame(df)

# Convert every column to proper factor
df[] <- lapply(df, function(x) factor(as.character(x)))

library(bnlearn)
library(e1071)

# ============================================================
# 1. Learn BN among courses (V1–V8)
# ============================================================

course_vars <- setdiff(names(df), "V9")   # V9 = internship/target column
courses_df <- df[, course_vars]

set.seed(123)
bn_courses <- hc(courses_df)
bn_courses

fitted_courses <- bn.fit(bn_courses, courses_df)

# Print CPTs
for (node in names(fitted_courses)) {
  cat("\nCPT for:", node, "\n")
  print(fitted_courses[[node]]$prob)
}

# ============================================================
# 2. Query: PH100 grade given EC100=DD, IT101=CC, MA101=CD
#    In your dataset:
#      PH100 = V6
#      EC100 = V1
#      IT101 = V3
#      MA101 = V5
# ============================================================

set.seed(2025)

samples <- cpdist(
  fitted_courses,
  nodes = "V6",                     # PH100
  evidence = (V1 == "DD" & 
                V3 == "CC" & 
                V5 == "CD"),
  n = 20000
)

cat("\nProbability distribution for PH100 (V6):\n")
print(prop.table(table(samples$V6)))

# ============================================================
# 3. Naive Bayes Classifier (independent features)
# ============================================================

set.seed(42)
n_repeats <- 20
acc_nb <- numeric(n_repeats)

for (i in 1:n_repeats) {
  
  idx <- sample.int(nrow(df))
  train_n <- floor(0.7 * nrow(df))
  
  train <- df[idx[1:train_n], ]
  test  <- df[idx[(train_n + 1):nrow(df)], ]
  
  # model
  nb_model <- naiveBayes(V9 ~ ., data = train)
  preds <- predict(nb_model, newdata = test)
  
  # accuracy
  acc_nb[i] <- mean(preds == test$V9)
  cat("NB Run", i, "Accuracy =", acc_nb[i], "\n")
}

cat("\nNaive Bayes Mean Accuracy =", mean(acc_nb),
    "\nSD =", sd(acc_nb), "\n")

# ============================================================
# 4. BN Classifier (dependencies allowed)
#    FIXED with factor-level alignment
# ============================================================

acc_bn <- numeric(n_repeats)

for (i in 1:n_repeats) {
  
  idx <- sample.int(nrow(df))
  train_n <- floor(0.7 * nrow(df))
  
  train <- df[idx[1:train_n], ]
  test  <- df[idx[(train_n + 1):nrow(df)], ]
  
  # IMPORTANT FIX: Ensure factor levels same in train & test
  for (col in names(df)) {
    train[[col]] <- factor(train[[col]], levels = levels(df[[col]]))
    test[[col]]  <- factor(test[[col]],  levels = levels(df[[col]]))
  }
  
  # Learn BN on training data
  bn_model <- hc(train)
  fitted_bn <- bn.fit(bn_model, train)
  
  # Predict V9 (internship)
  preds <- predict(fitted_bn,
                   node = "V9",
                   data = test,
                   method = "bayes-lw",
                   n = 2000)
  
  acc_bn[i] <- mean(preds == test$V9)
  cat("BN Run", i, "Accuracy =", acc_bn[i], "\n")
}

cat("\nBN Classifier Mean Accuracy =", mean(acc_bn),
    "\nSD =", sd(acc_bn), "\n")

