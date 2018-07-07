# System Function
# Aborting RSession

sesAb <- function(){
  makeActiveBinding("refresh", function() { shell("Rgui"); q("no") }, .GlobalEnv)
  makeActiveBinding("refresh", function() { system("R"); q("no") }, .GlobalEnv)
  refresh}
 
library(data.table)
library(dplyr)
library(caret)
library(MLmetrics)

gc()
multi_full_inactive <- fread("C:\\Users\\User\\Documents\\prinat_multiscan\\multi_full_inactive.csv", dec=",")
dim(multi_full_inactive)
str(multi_full_inactive)
str(multi_full_inactive$target)
names <- data.table(names(multi_full_inactive))

mcc_groups <- fread("D:\\BI\\mltsc\\mcc_groups_trans_num.csv", dec = ",", stringsAsFactors = F)
sd_features <- fread("D:\\BI\\mltsc\\soc_dem.csv", dec = ",")
pos_debit <- fread("D:\\BI\\mltsc\\pos_debit.csv", dec = ",")
deposit_credit <- fread("D:\\BI\\mltsc\\deposit_credit.csv", dec=",")
cash_debit <- fread("D:\\BI\\mltsc\\cash_debit.csv", dec = ",")

str(mcc_groups)
str(sd_features)
str(pos_debit)
str(deposit_credit)
str(cash_debit)

multi_full_inactive$id_resp <- as.integer(multi_full_inactive$id_resp); str(multi_full_inactive$id_resp)
mcc_groups$id_resp <- as.integer(mcc_groups$id_resp); str(mcc_groups$id_resp)
sd_features$id_resp <- as.integer(sd_features$id_resp); str(sd_features$id_resp)
pos_debit$id_resp <- as.integer(pos_debit$id_resp); str(pos_debit$id_resp)
deposit_credit$id_resp <- as.integer(deposit_credit$id_resp); str(deposit_credit$id_resp)
cash_debit$id_resp <- as.integer(cash_debit$id_resp); str(cash_debit$id_resp)

d1 <- dim(mcc_groups)
d2 <- dim(sd_features)
d3 <- dim(pos_debit)
d4 <- dim(deposit_credit)

sum(d1[2], d2[2], d3[2], d4[2])

multi_full_inactive <- merge(multi_full_inactive, mcc_groups, by = "id_resp", all.x = T)
multi_full_inactive <- merge(multi_full_inactive, sd_features, by = "id_resp", all.x = T)
multi_full_inactive <- merge(multi_full_inactive, pos_debit, by = "id_resp", all.x = T)
multi_full_inactive <- merge(multi_full_inactive, deposit_credit, by = "id_resp", all.x = T)
multi_full_inactive <- merge(multi_full_inactive, cash_debit, by = "id_resp", all.x = T)

dim(multi_full_inactive)
str(multi_full_inactive)

multi_full_inactive_copy <- copy(multi_full_inactive)
write.csv2(multi_full_inactive, "multi_full_inactive.csv")

multi_full_inactive[, target:=factor(ifelse(target=="Macro1", "Yes", "No"), levels = c("Yes", "No"))]
rm(cash_debit, deposit_credit, mcc_groups, pos_debit, sd_features)
multi_full_inactive[is.na(multi_full_inactive)] <- 0
near0vars <- nearZeroVar(multi_full_inactive[, -"id_resp", with=F], freqCut = 99 / 1, allowParallel = T, saveMetrics = T)
n0vars_to_drop <- row.names(near0vars[near0vars$nzv==T, ])
multi_full_inactive <- multi_full_inactive[, -n0vars_to_drop, with=F];
dim(multi_full_inactive)
rm(n0vars_to_drop, near0vars)

set.seed(42)
trainIndex <- createDataPartition(multi_full_inactive$target, p = .7, list = F)
train_multi <- multi_full_inactive[trainIndex[, 1], ]
test_multi <- multi_full_inactive[-trainIndex[, 1], ]
rm(trainIndex)
train_multi[, .N, target]
test_multi[, .N, target]
names_vars <- names(multi_full_inactive)
form <- paste0("target ~ ", paste0(names_vars, collapse = " + "))

cctrl1 <- trainControl(method = "cv", 
                       number = 3, 
                       sampling = "up",
                       classProbs = T, 
                       search = "random",
                       returnResamp = "all",  
                       savePredictions = "all"
)

set.seed(42)
fit.log1 <- caret::train(
  form = as.formula(form),
  data = train_multi,
  method = "regLogistic", 
  tuneLength = 10,
  metric = "Accuracy",
  preProc = c("center", "scale"),
  trControl = cctrl1)
  
pred1 <- predict(fit.log1, test_multi)
Precision(test_multi$target, factor(pred1)) 
Recall(test_multi$target, factor(pred1)) 
F1_Score(test_multi$target, factor(pred1))
Accuracy(test_multi$target, factor(pred1))
tabletest1 <- confusionMatrix(pred1, test_multi$target)
v_imp <- varImp(fit.log1)

form1 <- "target ~ mcc_4829_amount + pos_amount_sum_20 + mcg_20_amount +
Pos_Trx_Amount_sum + ats + pos_amount_max_20 + pos_amount_mean_20 +
mcc_4829_trx + pos_active_month + dep_amount_sum_20 + dep_amount_max_20 +
pos_amount_min_20 + pos_amount_mean_16 + mcg_20_trx + Transaction_Amount_Ekv_sum_sum +
pos_amount_min_16 + dep_amount_mean_20 + pos_amount_mean_3 + mcc_4829 + mcg_3"

set.seed(42)
fit.log1.1 <- caret::train(
  form = as.formula(form1),
  data = train_multi,
  method = "regLogistic", 
  tuneLength = 10,
  metric = "Accuracy",
  preProc = c("center", "scale"),
  trControl = cctrl1)
  
pred1.1 <- predict(fit.log1.1, test_multi)
Precision(test_multi$target, factor(pred1.1))
Recall(test_multi$target, factor(pred1.1)) 
F1_Score(test_multi$target, factor(pred1.1))
Accuracy(test_multi$target, factor(pred1.1))


pred1.1_prob <- predict(fit.log1.1, train_multi, type='prob')
train_with_prob <- cbind(train_multi, pred1.1_prob)
train_with_prob[target=="Yes", resid:=1-Yes]
train_with_prob[target!="Yes", resid:=Yes]
train <- train_with_prob[, .(id_resp, resid)]
train <- train[order(-resid)]
train[, cumsum_resid:=seq_len(.N)/.N]
new_train <- merge(train_with_prob, train, by = "id_resp")
new_train <- new_train[cumsum_resid > 0.05]

set.seed(42)
fit.log1.2 <- caret::train(
 form = as.formula(form1),
  data = new_train,
  method = "regLogistic", 
  tuneLength = 10,
  metric = "Accuracy",
  preProc = c("center", "scale"),
  trControl = cctrl1)
  
pred1.2 <- predict(fit.log1.2, test_multi)
Precision(test_multi$target, factor(pred1.2))
Recall(test_multi$target, factor(pred1.2))
F1_Score(test_multi$target, factor(pred1.2))
Accuracy(test_multi$target, factor(pred1.2))
pred1.2_prob <- predict(fit.log1.2, test_multi, type='prob')

t1 <- as.data.table(pred1.2_prob)
t1 <- cbind(test_multi$id_resp, t1)
t1 <- t1[, 1:2]

dim(t1)
names(t1) <- c("id_resp", "Yes")

multi_full_inactive_for_merge <- multi_full_inactive[, c("id_resp", "target")]
t1new <- merge(t1, multi_full_inactive_for_merge, by="id_resp", all.X = T)
dim(t1new)
write.csv2(t1new, "t1.csv")
saveRDS(fit.log1.2, "inactive_1segm_model.rds")
