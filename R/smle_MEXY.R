#' Performs efficient semiparametric estimation for general two-phase measurement error models when there are errors in both the outcome and covariates.
#'
#' @param Y_tilde Specifies the column of the error-prone outcome that is continuous. Subjects with missing values of \code{Y_tilde} are omitted from the analysis. This argument is required.
#' @param Y Specifies the column that stores the validated value of \code{Y_tilde} in the second phase. Subjects with missing values of \code{Y} are considered as those not selected in the second phase. This argument is required.
#' @param X_tilde Specifies the columns of the error-prone covariates. Subjects with missing values of \code{X_tilde} are omitted from the analysis. This argument is required.
#' @param X Specifies the columns that store the validated values of \code{X_tilde} in the second phase. Subjects with missing values of \code{X} are considered as those not selected in the second phase. This argument is required.
#' @param Bspline Specifies the columns of the B-spline basis. Subjects with missing values of \code{Bspline} are omitted from the analysis. This argument is required. 
#' @param Z Specifies the columns of the accurately measured covariates. Subjects with missing values of \code{Z} are omitted from the analysis. This argument is optional. 
#' @param data Specifies the name of the dataset. This argument is required.
#' @param hn_scale Specifies the scale of the perturbation constant in the variance estimation. For example, if \code{hn_scale = 0.5}, then the perturbation constant is \eqn{0.5n^{-1/2}}, where \eqn{n} is the first-phase sample size. The default value is \code{1}. This argument is optional.
#' @param MAX_ITER Specifies the maximum number of iterations in the EM algorithm. The default number is \code{2000}. This argument is optional.
#' @param TOL Specifies the convergence criterion in the EM algorithm. The default value is \code{1E-4}. This argument is optional.
#' @param noSE If \code{TRUE}, then the variances of the parameter estimators will not be estimated. The default value is \code{FALSE}. This argument is optional.
#' @param verbose If \code{TRUE}, then show details of the analysis. The default value is \code{FALSE}.
#' @param standardize If \code{TRUE}, then standardize the outcome and covariates during computation, which may help stabilizing standard error estimation. The default value is \code{FALSE}.
#' @return
#' \item{coefficients}{Stores the analysis results.}
#' \item{covariance}{Stores the covariance matrix of the regression coefficient estimates.}
#' \item{converge}{In parameter estimation, if the EM algorithm converges, then \code{converge = TRUE}. Otherwise, \code{converge = FALSE}.}
#' \item{converge2}{In variance estimation, if the EM algorithm converges, then \code{converge2 = TRUE}. Otherwise, \code{converge2 = FALSE}.}
#' @useDynLib TwoPhaseReg
#' @importFrom Rcpp evalCpp
#' @importFrom stats pchisq
#' @exportPattern "^[[:alpha:]]+"
smle_MEXY <- function (Y_tilde=NULL, Y=NULL, X_tilde=NULL, X=NULL, Z=NULL, Bspline=NULL, data=NULL, hn_scale=1, MAX_ITER=2000, TOL=1E-4, noSE=FALSE, verbose=FALSE, standardize=FALSE) {

    ###############################################################################################################
    #### check data ###############################################################################################
    storage.mode(MAX_ITER) = "integer"
	storage.mode(TOL) = "double"
	storage.mode(noSE) = "integer"
	
	if (is.null(data)) {
	    stop("No dataset is provided!")
	}

	if (is.null(Y_tilde)) {
		stop("The error-prone response Y_tilde is not specified!")
	} else {
		vars_ph1 = Y_tilde
	}

	if (is.null(X_tilde)) {
		stop("The error-prone covariates X_tilde is not specified!")
	} else {
		vars_ph1 = c(vars_ph1, X_tilde)
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
	
	if (length(X_tilde) != length(X)) {
	    stop("The number of columns in X_tilde and X is different!")
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
    	print(paste(length(id_exclude), "observations are excluded due to missing Y_tilde, X_tilde, or Z."))
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
    Y_tilde_vec = c(as.vector(data[-id_phase1,Y_tilde]), as.vector(data[id_phase1,Y_tilde]))
	storage.mode(Y_tilde_vec) = "double"
	
	X_tilde_mat = rbind(as.matrix(data[-id_phase1,X_tilde]), as.matrix(data[id_phase1,X_tilde]))
	storage.mode(X_tilde_mat) = "double"
	
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
	
	if (standardize == TRUE) {
	    my = mean(Y_tilde_vec)
	    sy = sd(Y_tilde_vec)
	    Y_tilde_vec = (Y_tilde_vec-my)/sy
	    Y_vec = (Y_vec-my)/sy
	    
	    mx = colMeans(X_tilde_mat)
	    sx = apply(X_tilde_mat, 2, sd)
	    for (i in 1:X_nc) {
	        X_tilde_mat[,i] = (X_tilde_mat[i]-mx[i])/sx[i]
	        X_mat[,i] = (X_mat[i]-mx[i])/sx[i]
	    }
	    
	    if (!is.null(Z)) {
	        mz = colMeans(Z_mat[,-1])
	        sz = apply(Z_mat[,-1], 2, sd)
	        Z_nc = length(Z)
	        for (i in 2:(Z_nc+1)) {
	            Z_mat[,i] = (Z_mat[,i]-mz[i])/sz[i]
	        }
	    }
	}
	#### prepare analysis #########################################################################################
	###############################################################################################################
	

	
	###############################################################################################################
	#### analysis #################################################################################################
	res = .Call("TwoPhase_MLE0_MEXY", Y_tilde_vec, X_tilde_mat, Y_vec, X_mat, Z_mat, Bspline_mat, hn, MAX_ITER, TOL, noSE, package="TwoPhaseReg")
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
	res_final = list(coefficients=res_coefficients, covariance=res_cov, converge=!res$flag_nonconvergence, converge2=!res$flag_nonconvergence_cov)
	res_final
    #### return results ###########################################################################################
    ###############################################################################################################
}
