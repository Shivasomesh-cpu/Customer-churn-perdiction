# StaySignal analysis functions. Statistical calculations use R's base stats package.

read_project_csv <- function(path) {
  raw <- utils::read.csv(
    path,
    header = TRUE,
    colClasses = "character",
    stringsAsFactors = FALSE,
    check.names = FALSE,
    strip.white = TRUE,
    na.strings = c("", "NA", "N/A", "NULL", "null"),
    fileEncoding = "UTF-8-BOM"
  )

  if (nrow(raw) == 0L || ncol(raw) < 2L) {
    stop("The CSV needs at least one data row and two columns.")
  }

  original_names <- names(raw)
  cleaned_names <- trimws(original_names)
  missing_names <- is.na(cleaned_names) | cleaned_names == ""
  cleaned_names[missing_names] <- paste0("Column ", which(missing_names))
  names(raw) <- make.unique(cleaned_names, sep = "_")

  warnings <- character()
  if (anyDuplicated(cleaned_names)) {
    warnings <- c(warnings, "Duplicate column names were made unique so every field can be analyzed.")
  }

  id_pattern <- "(^id$|_id$|id$|customer|account|phone|postal|zip|email|name$|code$)"
  for (column in names(raw)) {
    values <- trimws(raw[[column]])
    values[values == ""] <- NA_character_
    observed <- values[!is.na(values)]

    looks_like_id <- grepl(id_pattern, tolower(column), perl = TRUE)
    numeric_values <- suppressWarnings(as.numeric(gsub("[$,]", "", observed)))
    can_be_numeric <- length(observed) > 0L && all(!is.na(numeric_values)) && !looks_like_id

    if (can_be_numeric) {
      converted <- rep(NA_real_, length(values))
      keep <- !is.na(values)
      converted[keep] <- suppressWarnings(as.numeric(gsub("[$,]", "", values[keep])))
      raw[[column]] <- converted
    } else {
      raw[[column]] <- values
    }
  }

  rownames(raw) <- NULL
  list(data = raw, warnings = warnings)
}

column_profile <- function(data) {
  rows <- lapply(names(data), function(column) {
    values <- data[[column]]
    missing_n <- sum(is.na(values))
    observed <- values[!is.na(values)]
    data.frame(
      Variable = column,
      Type = if (is.numeric(values)) "Numeric" else "Categorical",
      Observed = length(observed),
      Missing = missing_n,
      `Missing %` = if (length(values)) 100 * missing_n / length(values) else 0,
      Distinct = length(unique(as.character(observed))),
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

numeric_summary <- function(data) {
  columns <- names(data)[vapply(data, is.numeric, logical(1))]
  if (!length(columns)) return(data.frame())

  rows <- lapply(columns, function(column) {
    values <- data[[column]]
    values <- values[is.finite(values)]
    n <- length(values)
    mean_value <- if (n) mean(values) else NA_real_
    sd_value <- if (n > 1L) stats::sd(values) else NA_real_
    margin <- if (n > 1L) stats::qt(.975, df = n - 1L) * sd_value / sqrt(n) else NA_real_
    data.frame(
      Variable = column,
      N = n,
      Missing = sum(!is.finite(data[[column]])),
      Mean = mean_value,
      SD = sd_value,
      Median = if (n) stats::median(values) else NA_real_,
      IQR = if (n) stats::IQR(values) else NA_real_,
      Minimum = if (n) min(values) else NA_real_,
      Maximum = if (n) max(values) else NA_real_,
      `Mean CI low` = mean_value - margin,
      `Mean CI high` = mean_value + margin,
      check.names = FALSE
    )
  })
  do.call(rbind, rows)
}

distribution_summary <- function(data, variable) {
  values <- data[[variable]]
  if (is.numeric(values)) {
    observed <- values[is.finite(values)]
    n <- length(observed)
    mean_value <- if (n) mean(observed) else NA_real_
    sd_value <- if (n > 1L) stats::sd(observed) else NA_real_
    margin <- if (n > 1L) stats::qt(.975, n - 1L) * sd_value / sqrt(n) else NA_real_
    return(data.frame(
      Measure = c("Observed", "Missing", "Mean", "Standard deviation", "Median", "IQR", "Minimum", "Maximum", "95% CI for mean"),
      Value = c(
        as.character(n), as.character(sum(!is.finite(values))),
        if (n) sprintf("%.3f", mean_value) else "—",
        if (n > 1L) sprintf("%.3f", sd_value) else "—",
        if (n) sprintf("%.3f", stats::median(observed)) else "—",
        if (n) sprintf("%.3f", stats::IQR(observed)) else "—",
        if (n) sprintf("%.3f", min(observed)) else "—",
        if (n) sprintf("%.3f", max(observed)) else "—",
        if (n > 1L) sprintf("[%.3f, %.3f]", mean_value - margin, mean_value + margin) else "Not enough data"
      ),
      stringsAsFactors = FALSE
    ))
  }

  labels <- as.character(values)
  labels[is.na(labels)] <- "(Missing)"
  counts <- sort(table(labels), decreasing = TRUE)
  data.frame(
    Category = names(counts),
    Count = as.integer(counts),
    `Share of rows` = as.integer(counts) / length(values),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

with_local_seed <- function(seed, code) {
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) old_seed <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(seed)
  force(code)
}

association_result <- function(data, first, second) {
  if (identical(first, second)) stop("Choose two different variables.")
  x <- data[[first]]
  y <- data[[second]]
  keep <- stats::complete.cases(x, y)
  x <- x[keep]
  y <- y[keep]
  n <- length(x)
  if (n < 3L) stop("Fewer than three complete observations are available for this comparison.")

  if (is.numeric(x) && is.numeric(y)) {
    if (length(unique(x)) < 2L || length(unique(y)) < 2L) stop("Correlation is undefined for a constant variable.")
    pearson <- stats::cor.test(x, y, method = "pearson")
    spearman <- stats::cor.test(x, y, method = "spearman", exact = FALSE)
    result <- data.frame(
      Test = c("Pearson correlation", "Spearman rank correlation"),
      N = n,
      Estimate = c(unname(pearson$estimate), unname(spearman$estimate)),
      `Effect measure` = c("r", "rho"),
      `95% CI` = c(
        if (!is.null(pearson$conf.int)) sprintf("[%.3f, %.3f]", pearson$conf.int[1], pearson$conf.int[2]) else "—",
        "— (rank test)"
      ),
      `p-value` = c(pearson$p.value, spearman$p.value),
      check.names = FALSE
    )
    return(list(rows = result, note = "Correlation describes association, not a causal effect."))
  }

  if (xor(is.numeric(x), is.numeric(y))) {
    numeric_values <- if (is.numeric(x)) x else y
    group_values <- if (is.numeric(x)) y else x
    group <- droplevels(factor(group_values))
    counts <- table(group)
    if (length(counts) < 2L) stop("The categorical variable needs at least two observed groups.")
    if (any(counts < 2L)) stop("Each group needs at least two observed numeric values.")

    if (length(counts) == 2L) {
      test <- stats::t.test(numeric_values ~ group, var.equal = FALSE)
      group_means <- tapply(numeric_values, group, mean)
      group_vars <- tapply(numeric_values, group, stats::var)
      pooled_sd <- sqrt(sum((counts - 1) * group_vars) / (sum(counts) - 2))
      cohen_d <- if (pooled_sd > 0) diff(group_means) / pooled_sd else NA_real_
      correction <- 1 - 3 / (4 * sum(counts) - 9)
      result <- data.frame(
        Test = "Welch two-sample t-test",
        N = n,
        Estimate = unname(diff(group_means)),
        `Effect measure` = "Hedges' g",
        Effect = cohen_d * correction,
        `95% CI` = sprintf("[%.3f, %.3f]", test$conf.int[1], test$conf.int[2]),
        `p-value` = test$p.value,
        check.names = FALSE
      )
      return(list(rows = result, note = "Mean difference is first observed group minus second; Welch's test does not assume equal variances."))
    }

    test <- stats::oneway.test(numeric_values ~ group, var.equal = FALSE)
    overall_mean <- mean(numeric_values)
    between <- sum(as.numeric(counts) * (as.numeric(tapply(numeric_values, group, mean)) - overall_mean)^2)
    total <- sum((numeric_values - overall_mean)^2)
    result <- data.frame(
      Test = "Welch one-way ANOVA",
      N = n,
      Estimate = unname(test$statistic),
      `Effect measure` = "F statistic",
      Effect = if (total > 0) between / total else NA_real_,
      `95% CI` = "—",
      `p-value` = test$p.value,
      check.names = FALSE
    )
    names(result)[names(result) == "Effect"] <- "Effect size (eta-squared)"
    return(list(rows = result, note = "Welch ANOVA compares group means without assuming equal variances; eta-squared is a descriptive effect size."))
  }

  x <- droplevels(factor(as.character(x)))
  y <- droplevels(factor(as.character(y)))
  tab <- table(x, y)
  if (nrow(tab) < 2L || ncol(tab) < 2L) stop("Both categorical variables need at least two observed levels.")
  pearson <- suppressWarnings(stats::chisq.test(tab, correct = FALSE))
  sparse <- any(pearson$expected < 5)
  if (all(dim(tab) == 2L) && sparse) {
    test <- stats::fisher.test(tab)
    method <- "Fisher exact test"
    statistic <- NA_real_
  } else if (sparse) {
    test <- with_local_seed(42, stats::chisq.test(tab, simulate.p.value = TRUE, B = 5000))
    method <- "Chi-square (Monte Carlo p-value)"
    statistic <- unname(pearson$statistic)
  } else {
    test <- pearson
    method <- "Pearson chi-square test"
    statistic <- unname(test$statistic)
  }
  denominator <- n * min(nrow(tab) - 1L, ncol(tab) - 1L)
  cramers_v <- if (denominator > 0) sqrt(unname(pearson$statistic) / denominator) else NA_real_
  result <- data.frame(
    Test = method,
    N = n,
    Estimate = statistic,
    `Effect measure` = "Cramer's V",
    Effect = cramers_v,
    `95% CI` = "—",
    `p-value` = test$p.value,
    check.names = FALSE
  )
  list(rows = result, note = "Cramer's V is a descriptive association measure; it does not indicate a causal relationship." )
}

association_screen <- function(data, target) {
  predictors <- setdiff(names(data), target)
  rows <- lapply(predictors, function(variable) {
    result <- tryCatch(association_result(data, target, variable), error = function(e) NULL)
    if (is.null(result) || !nrow(result$rows)) return(NULL)
    row <- result$rows[1, , drop = FALSE]
    data.frame(
      Variable = variable,
      Test = row$Test,
      N = row$N,
      Effect = if ("Effect" %in% names(row)) row$Effect else row$Estimate,
      `Effect measure` = if ("Effect measure" %in% names(row)) row$`Effect measure` else "Estimate",
      `p-value` = row$`p-value`,
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
  })
  rows <- Filter(Negate(is.null), rows)
  if (!length(rows)) return(data.frame())
  output <- do.call(rbind, rows)
  output$`BH q-value` <- stats::p.adjust(output$`p-value`, method = "BH")
  output <- output[order(output$`BH q-value`, output$`p-value`), , drop = FALSE]
  rownames(output) <- NULL
  output
}

stratified_indices <- function(outcome, train_share = 0.8, seed = 42L) {
  with_local_seed(seed, {
    split <- unlist(lapply(sort(unique(outcome)), function(level) {
      indices <- which(outcome == level)
      train_n <- min(length(indices) - 1L, max(1L, floor(length(indices) * train_share)))
      sample(indices, train_n)
    }), use.names = FALSE)
    list(train = sort(split), test = setdiff(seq_along(outcome), split))
  })
}

auc_rank <- function(actual, probability) {
  positive <- sum(actual == 1L)
  negative <- sum(actual == 0L)
  if (!positive || !negative) return(NA_real_)
  ranks <- rank(probability, ties.method = "average")
  (sum(ranks[actual == 1L]) - positive * (positive + 1) / 2) / (positive * negative)
}

fit_logistic_holdout <- function(data, outcome, event, predictors, threshold = 0.5) {
  predictors <- unique(setdiff(predictors, outcome))
  if (!length(predictors)) stop("Select at least one predictor.")
  outcome_values <- as.character(data[[outcome]])
  levels_outcome <- sort(unique(outcome_values[!is.na(outcome_values)]))
  if (length(levels_outcome) != 2L) stop("The selected outcome must have exactly two observed categories.")
  if (!event %in% levels_outcome) stop("Choose an event category that appears in the outcome column.")

  model_data <- data[, c(outcome, predictors), drop = FALSE]
  model_data <- model_data[stats::complete.cases(model_data), , drop = FALSE]
  model_data$.event <- as.integer(as.character(model_data[[outcome]]) == event)
  class_counts <- table(model_data$.event)
  if (length(class_counts) < 2L || any(class_counts < 5L)) {
    stop("At least five complete observations are needed in each outcome category for a stratified holdout.")
  }

  original_names <- c(outcome, predictors)
  safe_names <- make.names(original_names, unique = TRUE)
  names(model_data)[match(original_names, names(model_data))] <- safe_names
  outcome_name <- ".event"
  predictor_names <- safe_names[-1L]

  too_many_levels <- predictor_names[vapply(model_data[predictor_names], function(x) {
    !is.numeric(x) && length(unique(x)) > 30L
  }, logical(1))]
  if (length(too_many_levels)) {
    original <- original_names[match(too_many_levels, safe_names)]
    stop("Remove high-cardinality predictor(s) before fitting: ", paste(original, collapse = ", "), ".")
  }
  for (predictor in predictor_names) {
    if (!is.numeric(model_data[[predictor]])) model_data[[predictor]] <- factor(model_data[[predictor]])
  }

  split <- stratified_indices(model_data[[outcome_name]])
  train <- droplevels(model_data[split$train, , drop = FALSE])
  test <- model_data[split$test, , drop = FALSE]
  for (predictor in predictor_names) {
    if (is.factor(train[[predictor]])) {
      if (nlevels(train[[predictor]]) < 2L) stop("A selected categorical predictor has only one level in the training split: ", predictor)
      test[[predictor]] <- factor(as.character(test[[predictor]]), levels = levels(train[[predictor]]))
    }
  }

  formula <- stats::reformulate(predictor_names, response = outcome_name)
  captured_warnings <- character()
  fit <- withCallingHandlers(
    stats::glm(formula, data = train, family = stats::binomial()),
    warning = function(w) {
      captured_warnings <<- c(captured_warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  test <- test[stats::complete.cases(test[, c(outcome_name, predictor_names), drop = FALSE]), , drop = FALSE]
  if (!nrow(test)) stop("No held-out rows remain after handling categories absent from the training split.")
  probability <- as.numeric(stats::predict(fit, newdata = test, type = "response"))
  actual <- as.integer(test[[outcome_name]])
  predicted <- as.integer(probability >= threshold)
  true_positive <- sum(predicted == 1L & actual == 1L)
  true_negative <- sum(predicted == 0L & actual == 0L)
  false_positive <- sum(predicted == 1L & actual == 0L)
  false_negative <- sum(predicted == 0L & actual == 1L)
  metrics <- data.frame(
    Metric = c("Test rows", "ROC AUC", "Brier score", "Accuracy", "Sensitivity", "Specificity"),
    Value = c(
      as.character(nrow(test)),
      sprintf("%.3f", auc_rank(actual, probability)),
      sprintf("%.3f", mean((probability - actual)^2)),
      sprintf("%.1f%%", 100 * mean(predicted == actual)),
      if (sum(actual == 1L)) sprintf("%.1f%%", 100 * true_positive / sum(actual == 1L)) else "—",
      if (sum(actual == 0L)) sprintf("%.1f%%", 100 * true_negative / sum(actual == 0L)) else "—"
    ),
    stringsAsFactors = FALSE
  )

  coefficient_matrix <- summary(fit)$coefficients
  coefficient_rows <- which(!is.na(coefficient_matrix[, "Estimate"]))
  terms <- rownames(coefficient_matrix)[coefficient_rows]
  display_terms <- vapply(terms, function(term) {
    match_index <- which(vapply(predictor_names, function(name) startsWith(term, name), logical(1)))
    if (length(match_index)) {
      index <- match_index[which.max(nchar(predictor_names[match_index]))]
      paste0(original_names[index + 1L], substring(term, nchar(predictor_names[index]) + 1L))
    } else term
  }, character(1))
  estimate <- coefficient_matrix[coefficient_rows, "Estimate"]
  standard_error <- coefficient_matrix[coefficient_rows, "Std. Error"]
  coefficients <- data.frame(
    Term = display_terms,
    `Log-odds coefficient` = estimate,
    `Odds ratio` = exp(estimate),
    `OR CI low` = exp(estimate - 1.96 * standard_error),
    `OR CI high` = exp(estimate + 1.96 * standard_error),
    `p-value` = coefficient_matrix[coefficient_rows, ncol(coefficient_matrix)],
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  list(
    metrics = metrics,
    coefficients = coefficients,
    confusion = data.frame(
      `Observed / predicted` = c("No event", "Event"),
      `Predicted no event` = c(true_negative, false_negative),
      `Predicted event` = c(false_positive, true_positive),
      check.names = FALSE
    ),
    train_n = nrow(train),
    test_n = nrow(test),
    event = event,
    threshold = threshold,
    warnings = unique(captured_warnings)
  )
}
