#' Performs efficient semiparametric estimation for general two-phase measurement error models when the outcome is continuous and there are errors in both the outcome and covariates.
#'
#' @param Y_unval Specifies the column of the error-prone outcome that is continuous. Subjects with missing values of \code{Y_unval} are omitted from the analysis. This argument is required.
#' @param Y Specifies the column that stores the validated value of \code{Y_unval} in the second phase. Subjects with missing values of \code{Y} are considered as those not selected in the second phase. This argument is required.
#' @param X_unval Specifies the columns of the error-prone covariates. Subjects with missing values of \code{X_unval} are omitted from the analysis. This argument is required.
#' @param X Specifies the columns that store the validated values of \code{X_unval} in the second phase. Subjects with missing values of \code{X} are considered as those not selected in the second phase. This argument is required.
#' @param Bspline Specifies the columns of the B-spline basis. Subjects with missing values of \code{Bspline} are omitted from the analysis. This argument is required. 
#' @param Z Specifies the columns of the accurately measured covariates. Subjects with missing values of \code{Z} are omitted from the analysis. This argument is optional. 
#' @param data Specifies the name of the dataset. This argument is required.
#' @param hn_scale Specifies the scale of the perturbation constant in the variance estimation. For example, if \code{hn_scale = 0.5}, then the perturbation constant is \eqn{0.5n^{-1/2}}, where \eqn{n} is the first-phase sample size. The default value is \code{1}. This argument is optional.
#' @param MAX_ITER Specifies the maximum number of iterations in the EM algorithm. The default number is \code{2000}. This argument is optional.
#' @param TOL Specifies the convergence criterion in the EM algorithm. The default value is \code{1E-4}. This argument is optional.
#' @param noSE If \code{TRUE}, then the variances of the parameter estimators will not be estimated. The default value is \code{FALSE}. This argument is optional.
#' @param verbose If \code{TRUE}, then show details of the analysis. The default value is \code{FALSE}.
#' @return
#' \item{coefficients}{Stores the analysis results.}
#' \item{sigma}{Stores the residual standard error.}
#' \item{covariance}{Stores the covariance matrix of the regression coefficient estimates.}
#' \item{converge}{In parameter estimation, if the EM algorithm converges, then \code{converge = TRUE}. Otherwise, \code{converge = FALSE}.}
#' \item{converge2}{In variance estimation, if the EM algorithm converges, then \code{converge2 = TRUE}. Otherwise, \code{converge2 = FALSE}.}
#' @useDynLib TwoPhaseReg
#' @importFrom Rcpp evalCpp
#' @importFrom stats pchisq
#' @exportPattern "^[[:alpha:]]+"
smle_MEXY <- function (Y_unval=NULL, Y=NULL, X_unval=NULL, X=NULL, Z=NULL, Bspline=NULL, data=NULL, hn_scale=1, MAX_ITER=2000, TOL=1E-4, noSE=FALSE, verbose=FALSE) {

  ###############################################################################################################
  #### check data ###############################################################################################
  storage.mode(MAX_ITER) = "integer"
	storage.mode(TOL) = "double"
	storage.mode(noSE) = "integer"
	
	if (is.null(data)) {
	  stop("No dataset is provided!")
	}

	if (is.null(Y_unval)) {
		stop("The error-prone response Y_unval is not specified!")
	} else {
		vars_ph1 = Y_unval
	}

	if (is.null(X_unval)) {
		stop("The error-prone covariates X_unval is not specified!")
	} else {
		vars_ph1 = c(vars_ph1, X_unval)
	}

	if (is.null(Bspline)) {
    stop("The B-spline basis is not specified!")
	} else {
	  vars_ph1 = c(vars_ph1, Bspline)
	}

	if (is.null(Y)) {
		stop("The accurately measured response Y is not specified!")
	}
	
	if (is.null(X)) {
		stop("The validated covariates in the second-phase are not specified!")
	}
	
	if (length(X_unval) != length(X)) {
	  stop("The number of columns in X_unval and X is different!")
	}

	if (!is.null(Z)) {
		vars_ph1 = c(vars_ph1, Z)
	}
	
    id_exclude = c()
    for (var in vars_ph1) {
      id_exclude = union(id_exclude, which(is.na(data[,var])))
    }
	
	if (verbose) {
    print(paste("There are", nrow(data), "observations in the dataset."))
    print(paste(length(id_exclude), "observations are excluded due to missing Y_unval, X_unval, or Z."))
	}
	if (length(id_exclude) > 0) {
		data = data[-id_exclude,]
	}
	
  n = nrow(data)
	if (verbose) {
    print(paste("There are", n, "observations in the analysis."))
	}

    id_phase1 = which(is.na(data[,Y]))
    for (var in X) {
      id_phase1 = union(id_phase1, which(is.na(data[,var])))
    }
	if (verbose) {
		print(paste("There are", n-length(id_phase1), "observations validated in the second phase."))
	}
  #### check data ###############################################################################################
	###############################################################################################################

	
	
	###############################################################################################################
	#### prepare analysis #########################################################################################	
  Y_unval_vec = c(as.vector(data[-id_phase1,Y_unval]), as.vector(data[id_phase1,Y_unval]))
	storage.mode(Y_unval_vec) = "double"
	
	X_unval_mat = rbind(as.matrix(data[-id_phase1,X_unval]), as.matrix(data[id_phase1,X_unval]))
	storage.mode(X_unval_mat) = "double"
	
	Bspline_mat = rbind(as.matrix(data[-id_phase1,Bspline]), as.matrix(data[id_phase1,Bspline]))
	storage.mode(Bspline_mat) = "double"
	
	Y_vec = as.vector(data[-id_phase1,Y])
	storage.mode(Y_vec) = "double"
	
  X_mat = as.matrix(data[-id_phase1,X])
	storage.mode(X_mat) = "double"
	
  if (!is.null(Z)) {
    Z_mat = rbind(as.matrix(data[-id_phase1,Z]), as.matrix(data[id_phase1,Z]))
	  storage.mode(Z_mat) = "double"
  }
	
  cov_names = c("Intercept", X)
	if (!is.null(Z)) {
		cov_names = c(cov_names, Z)
	}
	
	ncov = length(cov_names)
	X_nc = length(X)
	rowmap = rep(NA, ncov)
	res_coefficients = matrix(NA, nrow=ncov, ncol=4)
	colnames(res_coefficients) = c("Estimate", "SE", "Statistic", "p-value")
	rownames(res_coefficients) = cov_names
	res_cov = matrix(NA, nrow=ncov, ncol=ncov)
	colnames(res_cov) = cov_names
	rownames(res_cov) = cov_names
	
	if (is.null(Z)) {
		Z_mat = rep(1., n)
		rowmap[1] = ncov
		rowmap[2:ncov] = 1:X_nc
	} else {
		Z_mat = cbind(1, Z_mat)
		rowmap[1] = X_nc+1
		rowmap[2:(X_nc+1)] = 1:X_nc
		rowmap[(X_nc+2):ncov] = (X_nc+2):ncov
	}
	
	hn = hn_scale/sqrt(n)
	#### prepare analysis #########################################################################################
	###############################################################################################################
	

	
	###############################################################################################################
	#### analysis #################################################################################################
	res = .Call("TwoPhase_MLE0_MEXY", Y_unval_vec, X_unval_0mat, Y_vec, X_mat, Z_mat, Bspline_mat, hn, MAX_ITER, TOL, noSE, package="TwoPhaseReg")
  #### analysis #################################################################################################
	###############################################################################################################
	
	

  ###############################################################################################################
  #### return results ###########################################################################################
 	res_coefficients[,1] = res$theta[rowmap]
	res_coefficients[which(res_coefficients[,1] == -999),1] = NA
	res_cov = res$cov_theta[rowmap, rowmap]
	res_cov[which(res_cov == -999)] = NA
	diag(res_cov)[which(diag(res_cov) < 0)] = NA
	
	res_coefficients[,2] = diag(res_cov)
	res_coefficients[which(res_coefficients[,2] > 0),2] = sqrt(res_coefficients[which(res_coefficients[,2] > 0),2])
	
	id_NA = which(is.na(res_coefficients[,1]) | is.na(res_coefficients[,2]))
	if (length(id_NA) > 0)
	{
	  res_coefficients[-id_NA,3] = res_coefficients[-id_NA,1]/res_coefficients[-id_NA,2]
	  res_coefficients[-id_NA,4] = 1-pchisq(res_coefficients[-id_NA,3]^2, df=1)
	}
	else
	{
	  res_coefficients[,3] = res_coefficients[,1]/res_coefficients[,2]
	  res_coefficients[,4] = 1-pchisq(res_coefficients[,3]^2, df=1)
	}
	res_final = list(coefficients=res_coefficients, sigma=sqrt(res$sigma_sq), covariance=res_cov, converge=!res$flag_nonconvergence, converge2=!res$flag_nonconvergence_cov)
	res_final
  #### return results ###########################################################################################
  ###############################################################################################################
}



#' Performs cross-validation to calculate the average predicted log likelihood for the \code{smle_MEXY} method. This function can be used to select the B-spline basis that yields the largest average predicted log likelihood.
#'
#' @param Y_unval Specifies the column of the error-prone outcome that is continuous. Subjects with missing values of \code{Y_unval} are omitted from the analysis. This argument is required.
#' @param Y Specifies the column that stores the validated value of \code{Y_unval} in the second phase. Subjects with missing values of \code{Y} are considered as those not selected in the second phase. This argument is required.
#' @param X_unval Specifies the columns of the error-prone covariates. Subjects with missing values of \code{X_unval} are omitted from the analysis. This argument is required.
#' @param X Specifies the columns that store the validated values of \code{X_unval} in the second phase. Subjects with missing values of \code{X} are considered as those not selected in the second phase. This argument is required.
#' @param Bspline Specifies the columns of the B-spline basis. Subjects with missing values of \code{Bspline} are omitted from the analysis. This argument is required.
#' @param Z Specifies the columns of the accurately measured covariates. Subjects with missing values of \code{Z} are omitted from the analysis. This argument is optional.
#' @param data Specifies the name of the dataset. This argument is required.
#' @param nfolds Specifies the number of cross-validation folds. The default value is \code{5}. Although \code{nfolds} can be as large as the sample size (leave-one-out cross-validation), it is not recommended for large datasets. The smallest value allowable is \code{3}.
#' @param MAX_ITER Specifies the maximum number of iterations in the EM algorithm. The default number is \code{2000}. This argument is optional.
#' @param TOL Specifies the convergence criterion in the EM algorithm. The default value is \code{1E-4}. This argument is optional.
#' @param verbose If \code{TRUE}, then show details of the analysis. The default value is \code{FALSE}.
#' @return
#' \item{avg_pred_loglike}{Stores the averge predicted log likelihood.}
#' \item{pred_loglike}{Stores the predicted log likelihoood in each fold.}
#' \item{converge}{Stores the convergence status of the EM algorithm in each run.}
#' @useDynLib TwoPhaseReg
#' @importFrom Rcpp evalCpp
#' @importFrom stats pchisq
#' @exportPattern "^[[:alpha:]]+"
cv_smle_MEXY <- function (Y_unval=NULL, Y=NULL, X_unval=NULL, X=NULL, Z=NULL, Bspline=NULL, data=NULL, nfolds=5, MAX_ITER=2000, TOL=1E-4, verbose=FALSE) {

  ###############################################################################################################
  #### check data ###############################################################################################
  storage.mode(MAX_ITER) = "integer"
  storage.mode(TOL) = "double"
  storage.mode(nfolds) = "integer"

  if (is.null(data)) {
    stop("No dataset is provided!")
  }

  if (is.null(Y_unval)) {
    stop("The error-prone response Y_unval is not specified!")
  } else {
    vars_ph1 = Y_unval
  }

  if (is.null(X_unval)) {
    stop("The error-prone covariates X_unval is not specified!")
  } else {
    vars_ph1 = c(vars_ph1, X_unval)
  }

  if (is.null(Bspline)) {
    stop("The B-spline basis is not specified!")
  } else {
    vars_ph1 = c(vars_ph1, Bspline)
  }

  if (is.null(Y)) {
    stop("The accurately measured response Y is not specified!")
  }

  if (is.null(X)) {
    stop("The validated covariates in the second-phase are not specified!")
  }

  if (length(X_unval) != length(X)) {
    stop("The number of columns in X_unval and X is different!")
  }

  if (!is.null(Z)) {
    vars_ph1 = c(vars_ph1, Z)
  }

  id_exclude = c()
  for (var in vars_ph1) {
    id_exclude = union(id_exclude, which(is.na(data[,var])))
  }

  if (verbose) {
    print(paste("There are", nrow(data), "observations in the dataset."))
    print(paste(length(id_exclude), "observations are excluded due to missing Y_unval, X_unval, or Z."))
  }
  if (length(id_exclude) > 0) {
    data = data[-id_exclude,]
  }

  n = nrow(data)
  if (verbose) {
    print(paste("There are", n, "observations in the analysis."))
  }

  id_phase1 = which(is.na(data[,Y]))
  for (var in X) {
    id_phase1 = union(id_phase1, which(is.na(data[,var])))
  }
  if (verbose) {
    print(paste("There are", n-length(id_phase1), "observations validated in the second phase."))
  }

  if (nfolds >= 3) {
    if (verbose) {
      print(paste0(nfolds, "-folds cross-validation will be performed."))
    }
  } else {
    stop("nfolds needs to be greater than or equal to 3!")
  }
  #### check data ###############################################################################################
  ###############################################################################################################



  ###############################################################################################################
  #### prepare analysis #########################################################################################
  Y_unval_vec = c(as.vector(data[-id_phase1,Y_unval]), as.vector(data[id_phase1,Y_unval]))
  storage.mode(Y_unval_vec) = "double"

  X_unval_mat = rbind(as.matrix(data[-id_phase1,X_unval]), as.matrix(data[id_phase1,X_unval]))
  storage.mode(X_unval_mat) = "double"

  Bspline_mat = rbind(as.matrix(data[-id_phase1,Bspline]), as.matrix(data[id_phase1,Bspline]))
  storage.mode(Bspline_mat) = "double"

  Y_vec = as.vector(data[-id_phase1,Y])
  storage.mode(Y_vec) = "double"

  X_mat = as.matrix(data[-id_phase1,X])
  storage.mode(X_mat) = "double"

  if (!is.null(Z)) {
    Z_mat = rbind(as.matrix(data[-id_phase1,Z]), as.matrix(data[id_phase1,Z]))
    storage.mode(Z_mat) = "double"
  }
  
  if (is.null(Z)) {
    Z_mat = rep(1., n)
  } else {
    Z_mat = cbind(1, Z_mat)
  }
  
  idx_fold = c(sample(1:nfolds, size = length(Y_vec), replace = TRUE),
               sample(1:nfolds, size = length(id_phase1), replace = TRUE))
  #### prepare analysis #########################################################################################
  ###############################################################################################################



  ###############################################################################################################
  #### analysis #################################################################################################
  pred_loglik = rep(NA, nfolds)
  converge = rep(NA, nfolds)
  for (fold in 1:nfolds) {
    Train = as.numeric(idx_fold != fold)
    res = .Call("TwoPhase_MLE0_MEXY_CV_loglik", Y_unval_vec, X_unval_mat, Y_vec, X_mat, Z_mat, Bspline_mat, MAX_ITER, TOL, Train, package = "TwoPhaseReg")
    pred_loglik[fold] = res$pred_loglike
    converge[fold] = !res$flag_nonconvergence
    if (pred_loglik[fold] == -999.) {
      pred_loglik[fold] = NA
    }
  }

  #### analysis #################################################################################################
  ###############################################################################################################



  ###############################################################################################################
  #### return results ###########################################################################################
  avg_pred_loglik = mean(pred_loglik, na.rm = TRUE)
  res_final = list(avg_pred_loglik=avg_pred_loglik, pred_loglik=pred_loglik, converge=converge)
  res_final  
  #### return results ###########################################################################################
  ###############################################################################################################
}

