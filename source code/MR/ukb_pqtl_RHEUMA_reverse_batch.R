# 批量反向两样本 MR：RHEUMA（暴露）-> UKB-PPP 蛋白（结局）
# 融合 2_step1.R / 2_step2.R 的批量流程：foreach、R2/F 过滤、MR、敏感性分析
#
# 可选环境变量：
#   RHEUMA_MR_OUTCOME_SOURCE=full|cis  默认 full；cis 用 cis_clump 小文件快速验证
#   RHEUMA_MR_PROTEINS=ICAM3,CXCL9     只跑指定蛋白，逗号分隔
#   RHEUMA_MR_RUN_PRESSO=TRUE|FALSE    默认 TRUE，跑 MR-PRESSO
#   RHEUMA_MR_SAVE_SENSITIVITY=TRUE|FALSE 默认 TRUE，保存敏感性分析
#   RHEUMA_MR_PRESSO_NB=1000          MR-PRESSO 置换次数，需大于工具变量数；默认 1000

suppressPackageStartupMessages(library(TwoSampleMR))
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(foreach))
suppressPackageStartupMessages(library(ggplot2))

base_dir <- "D:/BaiduNetdiskDownload/ukb_pqtl_MR"
exposure_file <- file.path(base_dir, "finngen_R11_M13_RHEUMA_expo_rt.txt")
protein_file <- file.path(base_dir, "rheuma_protein_outcomes.txt")

mode <- toupper(Sys.getenv("RHEUMA_MR_OUTCOME_SOURCE", unset = "full"))
selected_genes <- Sys.getenv("RHEUMA_MR_PROTEINS")
run_presso <- toupper(Sys.getenv("RHEUMA_MR_RUN_PRESSO", unset = "TRUE")) == "TRUE"
save_sensitivity <- toupper(Sys.getenv("RHEUMA_MR_SAVE_SENSITIVITY", unset = "TRUE")) == "TRUE"
presso_nb <- as.integer(Sys.getenv("RHEUMA_MR_PRESSO_NB", unset = "1000"))
if (is.na(presso_nb) || presso_nb < 1000) presso_nb <- 1000

if (mode == "CIS") {
  outcome_dir <- file.path(base_dir, "cis_clump")
  out_dir <- file.path(base_dir, "deCODE_backward_result", "RHEUMA_cis_test")
  mode <- "cis"
} else {
  outcome_dir <- file.path(base_dir, "snp")
  out_dir <- file.path(base_dir, "deCODE_backward_result")
  mode <- "full"
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# fread(cmd=...) 依赖 Rtools 自带 gzip/awk/head
Sys.setenv(PATH = paste0("C:/R/rtools45/usr/bin;", Sys.getenv("PATH")))

read_protein_list <- function() {
  if (file.exists(protein_file)) {
    genes <- trimws(readLines(protein_file, warn = FALSE))
  } else {
    genes <- c(
      "ICAM3", "MZB1", "CTSL", "TNFRSF9", "CXCL12", "UBE2L6", "SOD2",
      "IL12RB1", "RELT", "CKAP4", "CD22", "CD14", "IL2RA", "CD300E",
      "IFNLR1", "IL12B", "SIRPB1", "GCNT1", "TNFRSF10A", "LGMN", "MPO",
      "TFRC", "CD302", "CXCL16", "CD27", "TNFRSF6B", "CD83", "NPC2",
      "TNFRSF4", "CXCL9", "CCL21", "IL18BP"
    )
  }
  genes <- unique(genes[nzchar(genes)])
  if (nzchar(selected_genes)) {
    keep <- trimws(unlist(strsplit(selected_genes, ",", fixed = TRUE)))
    genes <- genes[toupper(genes) %in% toupper(keep)]
  }
  if (length(genes) == 0) stop("蛋白列表为空")
  genes
}

find_col_index <- function(hdr, choices) {
  for (ch in choices) if (ch %in% hdr) return(match(ch, hdr))
  NA_integer_
}

find_col_name <- function(hdr, choices) {
  for (ch in choices) if (ch %in% hdr) return(ch)
  NA_character_
}

find_outcome_file <- function(protein) {
  files <- Sys.glob(file.path(outcome_dir, paste0(protein, "_*")))
  if (mode == "full") {
    files <- files[grepl("\\.txt\\.gz$", files)]
    files <- files[!grepl("downloading|\\.baiduyun", files, ignore.case = TRUE)]
  } else {
    files <- files[grepl("_expo_rt\\.txt$", files)]
  }
  if (length(files) == 0) return(NA_character_)
  info <- file.info(files)
  files[which.max(info$mtime)]
}

read_outcome_header <- function(file) {
  if (mode == "full") {
    cmd <- paste("gzip -dc", shQuote(file, type = "cmd"), "| head -n 1")
    hdr_dt <- data.table::fread(cmd = cmd, sep = "\t", header = TRUE,
                                nrows = 0, showProgress = FALSE)
  } else {
    hdr_dt <- data.table::fread(file, sep = "\t", header = TRUE, nrows = 0,
                                showProgress = FALSE)
  }
  if (ncol(hdr_dt) == 0) stop("无法读取表头")
  names(hdr_dt)
}

# 只从大 gz 中抽取暴露 SNP 所在行，避免全量读入内存
# 注意：awk 只按 rsids2 过滤行，print $0 保留整行；BETA/SE/A1FREQ/LOG10P 等列会继续用于 format/harmonise/MR
read_outcome_rows <- function(file, snp_ids, awk_file) {
  snp_ids <- unique(tolower(trimws(snp_ids)))
  hdr <- read_outcome_header(file)
  snp_col_index <- find_col_index(hdr, c("rsids2", "rsids", "SNP", "ID"))
  if (is.na(snp_col_index)) stop("结局文件未找到 SNP 列")

  if (mode == "full") {
    ids_file <- tempfile(fileext = ".txt")
    writeLines(snp_ids, ids_file, useBytes = TRUE)
    on.exit(unlink(ids_file), add = TRUE)
    cmd <- paste0(
      "gzip -dc ", shQuote(file, type = "cmd"),
      " | awk -F \"\\t\" -v col=", snp_col_index, " -f ",
      shQuote(normalizePath(awk_file, winslash = "/", mustWork = TRUE), type = "cmd"),
      " ", shQuote(normalizePath(ids_file, winslash = "/", mustWork = TRUE), type = "cmd"),
      " -"
    )
    dt <- data.table::fread(cmd = cmd, sep = "\t", header = TRUE, fill = TRUE,
                            showProgress = FALSE)
  } else {
    dt <- data.table::fread(file, sep = "\t", header = TRUE, fill = TRUE,
                            showProgress = FALSE)
  }

  if (mode == "cis") {
    snp_col_name <- hdr[snp_col_index]
    dt <- dt[tolower(dt[[snp_col_name]]) %in% snp_ids, ]
  }
  dt
}

format_outcome <- function(dt, protein) {
  hdr <- names(dt)
  snp_col <- find_col_name(hdr, c("rsids2", "rsids", "SNP", "ID"))
  beta_col <- find_col_name(hdr, c("BETA", "Beta", "beta", "ES",
                                   "beta_site2", "beta_site1"))
  se_col <- find_col_name(hdr, c("SE", "se", "sebeta",
                                 "standard_error_site2", "standard_error_site1"))
  effect_col <- find_col_name(hdr, c("ALLELE1", "effectAllele",
                                     "effect_allele", "A1"))
  other_col <- find_col_name(hdr, c("ALLELE0", "otherAllele",
                                    "other_allele", "A0"))
  eaf_col <- find_col_name(hdr, c("A1FREQ", "EAF", "eaf"))
  p_col <- find_col_name(hdr, c("p", "P", "Pval", "pval",
                                "p_value_site2", "p_value_site1"))
  logp_col <- find_col_name(hdr, c("LOG10P", "LP", "log10p"))
  n_col <- find_col_name(hdr, c("N", "samplesize", "n",
                                "n_site2", "n_site1"))
  chr_col <- find_col_name(hdr, c("CHROM", "chr"))
  pos_col <- find_col_name(hdr, c("GENPOS", "POS", "pos"))

  if (is.na(snp_col) || is.na(beta_col) || is.na(se_col) || is.na(effect_col)) {
    stop("结局数据缺少 MR 必需列")
  }

  pval_col <- if (!is.na(p_col)) p_col else logp_col
  log_pval <- is.na(p_col) && !is.na(logp_col)

  out <- tryCatch(
    format_data(
      dat = as.data.frame(dt),
      type = "outcome",
      phenotype_col = "outcome",
      snp_col = snp_col,
      beta_col = beta_col,
      se_col = se_col,
      eaf_col = if (!is.na(eaf_col)) eaf_col else NULL,
      effect_allele_col = effect_col,
      other_allele_col = if (!is.na(other_col)) other_col else NULL,
      pval_col = if (!is.na(pval_col)) pval_col else NULL,
      samplesize_col = if (!is.na(n_col)) n_col else NULL,
      chr_col = if (!is.na(chr_col)) chr_col else NULL,
      pos_col = if (!is.na(pos_col)) pos_col else NULL,
      log_pval = log_pval
    ),
    error = function(e) NULL
  )
  if (is.null(out)) stop("format_data 失败")
  out$id.outcome <- protein
  out$outcome <- protein
  out
}

save_sensitivity_analysis <- function(harm_rt, filename, protein) {
  ok <- rep(TRUE, length(harm_rt$SNP))
  sens_note <- character()
  sens_error <- character()
  write_step <- function(file, label, fun) {
    tryCatch(fun(file),
             error = function(e) {
               sens_error <<- c(sens_error, paste0(label, ": ", conditionMessage(e)))
             })
  }

  pleio <- tryCatch(mr_pleiotropy_test(harm_rt), error = function(e) NULL)
  write_step(file.path(filename, "pleiotropy.txt"), "pleiotropy.txt",
             function(f) write.table(if (is.null(pleio)) data.frame(note = "SNP 数量不足或无法计算") else pleio,
                                     f, sep = "\t", row.names = FALSE, quote = FALSE))

  hete <- tryCatch(mr_heterogeneity(harm_rt), error = function(e) NULL)
  write_step(file.path(filename, "heterogeneity.txt"), "heterogeneity.txt",
             function(f) write.table(if (is.null(hete)) data.frame(note = "SNP 数量不足或无法计算") else hete,
                                     f, sep = "\t", row.names = FALSE, quote = FALSE))

  if (length(harm_rt$SNP) >= 2) {
    singlesnp_res <- tryCatch(mr_singlesnp(harm_rt), error = function(e) NULL)
    if (!is.null(singlesnp_res)) {
      singlesnp_or <- tryCatch(generate_odds_ratios(singlesnp_res),
                               error = function(e) singlesnp_res)
      write_step(file.path(filename, "singlesnpOR.csv"), "singlesnpOR.csv",
                 function(f) write.csv(singlesnp_or, f, row.names = FALSE))
      p2 <- tryCatch(mr_forest_plot(singlesnp_res), error = function(e) NULL)
      if (!is.null(p2)) write_step(file.path(filename, "forest.pdf"), "forest.pdf",
                                   function(f) ggsave(f, p2[[1]], width = 8, height = 8))
      p4 <- tryCatch(mr_funnel_plot(singlesnp_res), error = function(e) NULL)
      if (!is.null(p4)) write_step(file.path(filename, "funnelplot.pdf"), "funnelplot.pdf",
                                   function(f) ggsave(f, p4[[1]], width = 8, height = 8))
    } else {
      sens_note <- c(sens_note, "singlesnp/森林图/漏斗图: SNP 不足或无法计算")
    }

    leave_res <- tryCatch(mr_leaveoneout(harm_rt), error = function(e) NULL)
    if (!is.null(leave_res)) {
      write_step(file.path(filename, "leaveoneout.csv"), "leaveoneout.csv",
                 function(f) write.csv(leave_res, f, row.names = FALSE))
      p3 <- tryCatch(mr_leaveoneout_plot(leave_res), error = function(e) NULL)
      if (!is.null(p3)) write_step(file.path(filename, "sensitivity-analysis.pdf"), "sensitivity-analysis.pdf",
                                   function(f) ggsave(f, p3[[1]], width = 8, height = 8))
    } else {
      sens_note <- c(sens_note, "leaveoneout: SNP 不足或无法计算")
    }
  } else {
    sens_note <- c(sens_note, "SNP < 2，跳过大部分敏感性分析")
  }

  if (run_presso && length(harm_rt$SNP) >= 3) {
    presso_file <- file.path(filename, "presso.txt")
    if (nrow(harm_rt) >= presso_nb) {
      write_step(presso_file, "presso.txt",
                 function(f) writeLines(
                   paste0("MR-PRESSO 未运行：工具变量数 ", nrow(harm_rt),
                          " >= NbDistribution(", presso_nb, ")。可设置 RHEUMA_MR_PRESSO_NB 大于工具变量数后重跑；注意计算量会很大。"), f))
    } else {
      presso <- tryCatch(capture.output({
        run_mr_presso(harm_rt, NbDistribution = presso_nb)
      }), error = function(e) paste("MR-PRESSO 失败:", conditionMessage(e)))
      write_step(presso_file, "presso.txt",
                 function(f) writeLines(presso, f))
    }
  }

  if (length(sens_note) > 0) {
    write_step(file.path(filename, "sensitivity_skipped.txt"), "sensitivity_skipped.txt",
               function(f) writeLines(sens_note, f))
  }
  if (length(sens_error) > 0) {
    write_step(file.path(filename, "sensitivity_error.txt"), "sensitivity_error.txt",
               function(f) writeLines(sens_error, f))
  }
  }

proteins <- read_protein_list()
expo_rt <- read_exposure_data(
  filename = exposure_file,
  sep = "\t",
  phenotype_col = "exposure",
  snp_col = "SNP",
  beta_col = "BETA",
  se_col = "SE",
  eaf_col = "A1FREQ",
  effect_allele_col = "ALLELE1",
  other_allele_col = "ALLELE0",
  pval_col = "P",
  samplesize_col = "N",
  chr_col = "CHR",
  pos_col = "POS"
)
expo_rt$id.exposure <- "finngen_R11_M13_RHEUMA"
expo_rt$exposure <- "RHEUMA"

log_file <- file.path(out_dir, "process_log.txt")
awk_file <- NULL
if (mode == "full") {
  awk_file <- tempfile(fileext = ".awk")
  writeLines(
    c(
      "NR == FNR { ids[tolower($1)] = 1; next }",
      "FNR == 1 { print; next }",
      "{ if (tolower($col) in ids) print }"
    ),
    awk_file
  )
}
on.exit({ if (!is.null(awk_file)) unlink(awk_file) }, add = TRUE)

cat("暴露 SNP 数: ", nrow(expo_rt), "\n", sep = "")
cat("蛋白结局数: ", length(proteins), "\n", sep = "")
cat("输出目录: ", out_dir, "\n", sep = "")
cat("MR-PRESSO: ", run_presso, "\n", sep = "")
cat("保存敏感性分析: ", save_sensitivity, "\n", sep = "")
cat("MR-PRESSO NbDistribution: ", presso_nb, "\n", sep = "")

result_summary <- NULL
result_summary <- foreach(i = proteins, .combine = rbind,
                          .errorhandling = "pass") %do% {
  tryCatch({
    message("\n开始处理: ", i)
    protein_dir <- file.path(out_dir, i)
    if (!dir.exists(protein_dir)) dir.create(protein_dir, recursive = TRUE,
                                             showWarnings = FALSE)

    target_file <- find_outcome_file(i)
    if (is.na(target_file)) {
      message("结局文件未下载/未完成: ", i)
      return(data.frame(protein = i, nsnp = NA, method = NA, b = NA,
                        se = NA, pval = NA, OR = NA, OR_lci95 = NA,
                        OR_uci95 = NA, mean_f = NA, status = "no_file"))
    }

    outcome_raw <- read_outcome_rows(target_file, expo_rt$SNP, awk_file)
    message("read outcome rows: ", nrow(outcome_raw))
    if (nrow(outcome_raw) == 0) {
      message("结局中无共同 SNP: ", i)
      return(data.frame(protein = i, nsnp = NA, method = NA, b = NA,
                        se = NA, pval = NA, OR = NA, OR_lci95 = NA,
                        OR_uci95 = NA, mean_f = NA, status = "no_match"))
    }

    outc_rt <- format_outcome(outcome_raw, i)
    message("format outcome rows: ", nrow(outc_rt))
    harm_rt <- harmonise_data(exposure_dat = expo_rt, outcome_dat = outc_rt,
                              action = 2)
    harm_rt$id.outcome <- i
    harm_rt$outcome <- i
    harm_rt <- subset(harm_rt, mr_keep == TRUE)
    message("harmonise mr_keep rows: ", nrow(harm_rt))

    if (nrow(harm_rt) == 0) {
      message("harmonise 后无有效 SNP: ", i)
      return(data.frame(protein = i, nsnp = 0, method = NA, b = NA,
                        se = NA, pval = NA, OR = NA, OR_lci95 = NA,
                        OR_uci95 = NA, mean_f = NA, status = "no_harmonise"))
    }

    # 参考 2_step1/2_step2：计算 R2 和 F，仅保留 F>10 的工具变量
    harm_rt$R2 <- (2 * harm_rt$beta.exposure^2) /
      (2 * harm_rt$beta.exposure^2 +
         2 * harm_rt$samplesize.exposure * harm_rt$se.exposure^2)
    harm_rt$f <- harm_rt$R2 * (harm_rt$samplesize.exposure - 2) /
      (1 - harm_rt$R2)
    harm_rt$meanf <- mean(harm_rt$f)
    harm_rt <- harm_rt[harm_rt$f > 10, ]
    message("F>10 rows: ", nrow(harm_rt))

    if (nrow(harm_rt) == 0) {
      message("F>10 后无 SNP: ", i)
      return(data.frame(protein = i, nsnp = 0, method = NA, b = NA,
                        se = NA, pval = NA, OR = NA, OR_lci95 = NA,
                        OR_uci95 = NA, mean_f = NA, status = "f_le10"))
    }

    mr_result <- tryCatch(mr(harm_rt), error = function(e) NULL)
    if (is.null(mr_result) || nrow(mr_result) == 0) {
      message("MR 无有效结果: ", i)
      return(data.frame(protein = i, nsnp = nrow(harm_rt), method = NA,
                        b = NA, se = NA, pval = NA, OR = NA,
                        OR_lci95 = NA, OR_uci95 = NA,
                        mean_f = mean(harm_rt$f), status = "no_mr"))
    }

    result_or <- generate_odds_ratios(mr_result)
    message("OR result rows: ", nrow(result_or))
    result_or$protein <- i

    write.table(harm_rt, file.path(protein_dir, "harmonise.txt"),
                sep = "\t", row.names = FALSE, quote = FALSE)
    write.csv(result_or, file.path(protein_dir, "MR_results.csv"),
              row.names = FALSE)

    if (save_sensitivity) {
      tryCatch(save_sensitivity_analysis(harm_rt, protein_dir, i),
               error = function(e) {
                 writeLines(paste("敏感性分析失败:", conditionMessage(e)),
                            file.path(protein_dir, "sensitivity_failed.txt"))
               })
    }

    p1 <- tryCatch(mr_scatter_plot(result_or, harm_rt), error = function(e) NULL)
    if (!is.null(p1)) {
      ggsave(file.path(protein_dir, "scatter.pdf"), p1[[1]],
             width = 8, height = 8)
    }

    ivw <- result_or[grepl("Inverse variance weighted|IVW", result_or$method), ]
    ivw_idx <- which(grepl("Inverse variance weighted", result_or$method, ignore.case = TRUE) |
                      grepl("IVW", result_or$method, ignore.case = TRUE))
    ivw <- if (length(ivw_idx) > 0) result_or[ivw_idx[1], , drop = FALSE] else NULL
    or_col <- grep("^or$", names(result_or), ignore.case = TRUE)
    or_lci_col <- grep("^or_lci95$", names(result_or), ignore.case = TRUE)
    or_uci_col <- grep("^or_uci95$", names(result_or), ignore.case = TRUE)
    ivw_row <- data.frame(
      protein = i,
      nsnp = nrow(harm_rt),
      method = if (!is.null(ivw)) as.character(ivw$method[1]) else NA_character_,
      b = if (!is.null(ivw)) ivw$b[1] else NA_real_,
      se = if (!is.null(ivw)) ivw$se[1] else NA_real_,
      pval = if (!is.null(ivw)) ivw$pval[1] else NA_real_,
      OR = if (!is.null(ivw) && length(or_col) > 0) ivw[[or_col[1]]][1] else NA_real_,
      OR_lci95 = if (!is.null(ivw) && length(or_lci_col) > 0) ivw[[or_lci_col[1]]][1] else NA_real_,
      OR_uci95 = if (!is.null(ivw) && length(or_uci_col) > 0) ivw[[or_uci_col[1]]][1] else NA_real_,
      mean_f = mean(harm_rt$f),
      status = if (!is.null(ivw)) "done" else "no_ivw",
      stringsAsFactors = FALSE
    )
    message("完成: ", i, " (nsnp=", nrow(harm_rt), ")")
    ivw_row
  }, error = function(e) {
    message("处理失败: ", i, " -> ", conditionMessage(e))
    data.frame(protein = i, nsnp = NA, method = NA, b = NA,
               se = NA, pval = NA, OR = NA, OR_lci95 = NA,
               OR_uci95 = NA, mean_f = NA, status = "error")
  })
}

if (is.null(result_summary) || nrow(result_summary) == 0) {
  cat("没有可汇总的结果\n")
} else {
  write.csv(result_summary, file.path(out_dir, "MR_results_batch.csv"),
            row.names = FALSE)
  write.table(result_summary, file.path(out_dir, "MR_results_batch.txt"),
              sep = "\t", row.names = FALSE, quote = FALSE)
  sig <- result_summary[!is.na(result_summary$pval) &
                          result_summary$status == "done" &
                          result_summary$pval < 0.05, ]
  if (nrow(sig) > 0) {
    write.csv(sig, file.path(out_dir, "MR_IVW_pval0.05_results.csv"),
              row.names = FALSE)
    write.table(sig, file.path(out_dir, "MR_IVW_pval0.05_results.txt"),
                sep = "\t", row.names = FALSE, quote = FALSE)
  }
}

cat("\n批量反向 MR 完成！\n")
cat("结果目录: ", out_dir, "\n", sep = "")
