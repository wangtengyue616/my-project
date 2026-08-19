#--------------------------------
# RA分析：多暴露因素批量 Cox 分析脚本（含四个模型）
#--------------------------------
library(survival)
library(dplyr)
library(data.table)
library(tidyr)
library(splines)
#---并发包---
library(foreach)
library(doParallel)

#---------------------------------------------------------------
# COX回归基本流程
# 1.读取数据：
# 1）暴露因素csv文件exposure_file：即所有血浆蛋白数据，每列是一个血浆蛋白特征，代表一个暴露因素
# 2）结局因素csv文件outcome_file：即RA风险标签文件
# 3）协变量：包括了性别、年龄、运动等连续和分类变量

# 2.合并数据方便分析（可选）

# 3.协变量插补：连续变量使用均值插补，分类变量使用众数插补，并转换成factor因子

# 4.识别暴露变量类型并处理参考组（血浆蛋白数据都是连续变量，可省略）

# 5. 定义四个模型的协变量

# 6.循环读取暴露因素，进行cox分析：fml <- as.formula(paste("S ~", var, "+", paste(covs, collapse = " + ")))

# 7.保存结果
#---------------------------------------------------------------

# 设置工作路径
working_path <- "/work/data" 
setwd(working_path)

#-----------------------------
# Step 1. 读取数据
#-----------------------------
exposure_file <- "CBP_119.csv" # 暴露
outcome_file <- "code.csv" # 结局
covariates_file <- "RA_cox_cov1222.csv" # 协变量
# info_file <- "outcome-info.tsv"

dt <- fread(outcome_file)
exposures <- fread(exposure_file)
covariates <- fread(covariates_file)
# info <- fread(info_file, sep = "\t")

# 取三者交集
common_ids <- Reduce(intersect, list(dt$eid, exposures$eid, covariates$eid))
dt <- dt[eid %in% common_ids, ]
exposures <- exposures[eid %in% common_ids, ]
# info <- fread(info_file, sep = "\t")

#-----------------------------
# Step 2. 合并总表
#-----------------------------
dt <- merge(dt, exposures, by = "eid")
dt <- merge(dt, covariates, by = "eid")

#-----------------------------
# Step 3. 协变量插补 
# #-----------------------------
continuous_covs <- c("age", 'bmi',"tomson_index")
for(col in continuous_covs){
  if(col %in% colnames(dt)){
    dt[[col]][is.na(dt[[col]])] <- mean(dt[[col]], na.rm = TRUE)
  }
}
categorical_covs <- c("race", "somking", "drinking", "Education")
                      
for(col in categorical_covs){
  if(col %in% colnames(dt)){
    mode_val <- names(sort(table(dt[[col]]), decreasing = TRUE))[1]
    dt[[col]][is.na(dt[[col]])] <- mode_val
    dt[[col]] <- as.factor(dt[[col]])
  }
}

# 删除已确诊
dt <- dt[dt$early_illness_label != 1, ]
dt <- drop_na(dt, later_illness_label)

# 生存对象
dt$follow_up_time <- as.numeric(dt$follow_up_time)
S <- Surv(dt$follow_up_time, dt$later_illness_label == 1)

exposure_vars <- setdiff(colnames(exposures), "eid") ##
final_exposure_vars <- exposure_vars ##

write.csv(dt, paste0("CBP_dt", ".csv"), row.names = FALSE)
#-----------------------------
# Step 5. 定义模型的协变量
#-----------------------------
model_covs <- list(
  Model1 = NULL ,
   Model2 = c("age","sex","race"),  # 写入对应模型预设的协变量
   Model3 = c("age","sex","race","bmi")
)
print(dim(dt))
#-----------------------------
# Step 6. 批量 Cox 回归
#-----------------------------

# 保留 2 个核心给系统，其余全部用于计算
no_cores <- detectCores() - 2  
registerDoParallel(cores = no_cores)
message(paste("Core：", no_cores))

for(mod_name in names(model_covs)){
  covs <- model_covs[[mod_name]]
  
  message("Running：", mod_name, " ...")
  
  # .combine = bind_rows : 自动将每次循环的结果合并成一个大表格
  # .packages : 确保每个核都加载了 survival 库
  all_res <- foreach(var = final_exposure_vars, 
                     .combine = dplyr::bind_rows, 
                     .packages = c('survival', 'dplyr'),
                     .errorhandling = 'remove') %dopar% {
    
    # 检查数据有效性
    if (all(is.na(dt[[var]])) || length(unique(na.omit(dt[[var]]))) < 2) return(NULL)
    
    # 构建公式
    # fml_str <- paste0("Surv(follow_up_time, later_illness_label==1) ~ `", var, "` + ", paste(covs, collapse = " + "))
    # fml <- as.formula(fml_str)
    if (is.null(covs) || length(covs) == 0) {
      fml <- as.formula(
        paste0("Surv(follow_up_time, later_illness_label==1) ~ `", var, "`")
      )
    } else {
      fml <- as.formula(
        paste0(
          "Surv(follow_up_time, later_illness_label==1) ~ `",
          var, "` + ",
          paste(covs, collapse = " + ")
        )
      )
    }
    
    # 运行 Cox
    # model=FALSE, x=FALSE, y=FALSE 可以稍微减少内存占用和加速
    fit <- try(coxph(fml, data = dt, model = FALSE, x = FALSE, y = FALSE), silent = TRUE)
    
    if (inherits(fit, "try-error")) return(NULL)
    
    # 提取结果
    sm <- summary(fit)
    coef_df <- as.data.frame(sm$coefficients)
    coef_df$var <- rownames(coef_df)
    coef_df$exposure <- var
    coef_df$HR <- coef_df$`exp(coef)`
    coef_df$lower95 <- sm$conf.int[,"lower .95"]
    coef_df$upper95 <- sm$conf.int[,"upper .95"]
    
    # 提取事件数
    coef_df$n_total <- fit$n
    coef_df$n_events <- fit$nevent # coxph 对象直接包含 nevent，比 sum(y[,2]) 更快
    
    return(coef_df[, c("exposure","var","coef","HR","lower95","upper95",
                       "se(coef)","z","Pr(>|z|)","n_total","n_events")])
  }
  
  # 保存结果 
  write.csv(all_res, paste0("CBP_test", mod_name, ".csv"), row.names = FALSE)
  message("✅ ", mod_name, " finished. Results saved.")


  if (!is.null(all_res) && nrow(all_res) > 0) {

    ## ---------- 只保留 exposure == var ----------
    all_res <- all_res[all_res$exposure == all_res$var, ]

    ## ---------- Bonferroni / FDR 校正 ----------
    all_res$p_bonf <- NA
    all_res$p_fdr  <- NA

    valid_idx <- !is.na(all_res$`Pr(>|z|)`)

    if (any(valid_idx)) {

      # Bonferroni
      all_res$p_bonf[valid_idx] <- p.adjust(
        all_res$`Pr(>|z|)`[valid_idx],
        method = "bonferroni"
      )

      # FDR (BH)
      all_res$p_fdr[valid_idx] <- p.adjust(
        all_res$`Pr(>|z|)`[valid_idx],
        method = "BH"
      )
    }

    all_res$significant_bonf <- all_res$p_bonf < 0.05
    all_res$significant_fdr  <- all_res$p_fdr  < 0.05
  }

  #-----------------------------
  # 保存结果2
  #-----------------------------
  out_file <- paste0("CBP_test", mod_name, "_protein_bonf.csv")
  write.csv(all_res, out_file, row.names = FALSE)
  message("✅ ", mod_name, " finished. Protein-only results saved with correction.")

}

stopImplicitCluster()
