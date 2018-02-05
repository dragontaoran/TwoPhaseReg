#' Performs efficient semiparametric estimation for regression models under general two-phase designs.
#'
#' @param Y Specifies the column of the continuous, binary (\eqn{0} or \eqn{1}), or time-to-event outcomes. Subjects with missing values of \code{Y} are omitted from the analysis. This option is required.
#' @param L Specifies the column of left-truncation times. Missing values are set to \eqn{0}. This option is optional when performing Cox proportional hazards regression.
#' @param Delta Specifies the column of the event indicators. This option is required when performing Cox proportional hazards regression.
#' @param X Specifies the columns of expensive covariates. Subjects with missing values of \code{X} are considered as those not selected in the second phase. This argument is required.
#' @param Z Specifies the columns of the inexpensive covariates. Subjects with missing values of \code{Z} are omitted from the analysis. This argument is optional.
#' @param Bspline_Z Specifies the columns of the B-spline basis. Subjects with missing values of \code{Bspline_Z} are omitted from the analysis. This argument is optional.
#' @param data Specifies the name of the dataset. This argument is required.
#' @param hn_scale Specifies the scale of the perturbation constant in the variance estimation. For example, if \code{hn_scale = 0.5}, then the perturbation constant is \eqn{0.5n^{-1/2}}, where \eqn{n} is the first-phase sample size in the analysis. The default value is \code{1}. This argument is optional, and is not needed when there is no \code{Z}.
#' @param MAX_ITER Specifies the maximum number of iterations in the EM algorithm. The default number is \code{2000}. This argument is optional.
#' @param TOL Specifies the convergence criterion in the EM algorithm. The default value is \code{1E-4}. This argument is optional.
#' @param noSE If \code{TRUE}, then the variances of the parameter estimators will not be estimated. The default value is \code{FALSE}. This argument is optional.
#' @param model Specifies the model. Possible values are "linear", "logistic", and "coxph". The default value is "linear".
#' @param verbose If \code{TRUE}, then show details of the analysis. The default value is \code{FALSE}.
#' @return
#' \item{coefficients}{Stores the analysis results.}
#' \item{convergence}{In parameter estimation, if the EM algorithm converges, then \code{convergence = TRUE}. Otherwise, \code{convergence = FALSE}.}
#' \item{convergence_var}{In variance estimation, if the EM algorithm converges, then \code{convergence_cov = TRUE}. Otherwise, \code{convergence_cov = FALSE}.}
#' @examples
#' library(TwoPhaseReg)
#' n = 2000
#' n2 = 600
#' true_beta = 0.3
#' true_gamma = 0.4
#' true_eta = 0.5
#' seed = 12345
#' r = 0.3
#' N_SIEVE = 8
#'
#' #### Sieve with histogram bases
#' set.seed(12345)
#' simW = runif(n)
#' U2 = runif(n)
#' simX = runif(n)
#' simZ = r*simX+U2
#' simY = true_beta*simX+true_gamma*simZ+true_eta*simW+rnorm(n)
#' order.simY = order(simY)
#' phase2.id = c(order.simY[1:(n2/2)], order.simY[(n-(n2/2)+1):n])
#' Bspline_Z = matrix(NA, nrow=n, ncol=N_SIEVE)
#' cut_z = cut(simZ, breaks=quantile(simZ, probs=seq(0, 1, 1/N_SIEVE)), include.lowest = TRUE)
#' for (i in 1:N_SIEVE)
#' {
#'     Bspline_Z[,i] = as.numeric(cut_z == names(table(cut_z))[i])
#' }
#' colnames(Bspline_Z) = paste("bs", 1:N_SIEVE, sep="")
#' dat = data.frame(Y=simY, X=simX, Z=simZ, W=simW, Bspline_Z)
#' dat[-phase2.id,"X"] = NA
#' 
#' res = smle(Y="Y", X="X", Z=c("Z","W"), Bspline_Z=colnames(Bspline_Z), data=dat)
#' res
#'
#' #### Sieve with linear bases
#' library(splines)
#' set.seed(12345)
#' U1 = runif(n)
#' U2 = runif(n)
#' simX = runif(n)
#' simZ_1 = r*simX+U1
#' simZ_2 = r*simX+U2
#' simY = true_beta*simX+true_gamma*simZ_1+true_eta*simZ_2+rnorm(n)
#' order.simY = order(simY)
#' phase2.id = c(order.simY[1:(n2/2)], order.simY[(n-(n2/2)+1):n])
#' bs1 = bs(simZ_1, df=N_SIEVE, degree=1, Boundary.knots=range(simZ_1), intercept=TRUE)
#' bs2 = bs(simZ_2, df=N_SIEVE, degree=1, Boundary.knots=range(simZ_2), intercept=TRUE)
#' Bspline_Z = matrix(NA, ncol=N_SIEVE^2, nrow=n)
#' for (i in 1:ncol(bs1)) {
#'     for (j in 1:ncol(bs2)) {
#'         idx = i*N_SIEVE+j-N_SIEVE
#'         Bspline_Z[,idx] = bs1[,i]*bs2[,j]
#'     }
#' }
#' colnames(Bspline_Z) = paste("bs", 1:ncol(Bspline_Z), sep="")
#' dat = data.frame(Y=simY, X=simX, Z1=simZ_1, Z2=simZ_2, Bspline_Z)
#' dat[-phase2.id,"X"] = NA
#'
#' res = smle(Y="Y", X="X", Z=c("Z1", "Z2"), Bspline_Z=colnames(Bspline_Z), data=dat)
#' res
#' @references Tao, R., Zeng, D., and Lin, D. Y. (2017). "Efficient Semiparametric Inference Under Two-Phase Sampling, with Applications to Genetic Association Studies", Journal of the American Statistical Association, 112: 1468-1476.
#' @useDynLib TwoPhaseReg
#' @importFrom Rcpp evalCpp
#' @importFrom stats pchisq
#' @exportPattern "^[[:alpha:]]+"
smle <- function (Y=NULL, L=NULL, Delta=NULL, X=NULL, Z=NULL, W=NULL, Bspline_Z=NULL, data=NULL, hn_scale=1, MAX_ITER=2000,
    TOL=1E-4, noSE=FALSE, model="linear", verbose=FALSE) {

    ###############################################################################################################
    #### check data ###############################################################################################
    storage.mode(MAX_ITER) = "integer"
	storage.mode(TOL) = "double"
	storage.mode(noSE) = "integer"
	
	if (!(model %in% c("linear", "logistic", "coxph"))) {
	    stop("Model should be linear, logistic, or coxph!")
	}

	if (is.null(data)) {
	    stop("No dataset is provided!")
	}

	if (is.null(Y)) {
		stop("The response Y is not specified!")
	} else {
		vars_ph1 = Y
	}
	
	if (model == "coxph") {
		if (is.null(Delta)) {
			stop("The event indicator Delta is not specified!")
		} else {
			vars_ph1 = c(vars_ph1, Delta)
		}
	    if (!is.null(L)) {
	        data[which(is.na(data[,L])),L] = 0
	    }
	}

	if (is.null(X)) {
		stop("The expensive covariates X are not specified!")
	}

	if (!is.null(Z)) {
	    if (is.null(Bspline_Z)) {
	        stop("The B-spline functions Bspline_Z are not specified!")
	    } else {
    		vars_ph1 = c(vars_ph1, Z, Bspline_Z)
	    }
	}

    id_exclude = c()
    for (var in vars_ph1) {
        id_exclude = union(id_exclude, which(is.na(data[,var])))
    }
	
	if (verbose) {
    	print(paste("There are", nrow(data), "observations in the dataset."))
    	print(paste(length(id_exclude), "observations are excluded due to missing Y, Delta, Z, or Bspline_Z."))
	}
	if (length(id_exclude) > 0) {
		data = data[-id_exclude,]
	}
	
    n = nrow(data)
	if (verbose) {
    	print(paste("There are", n, "observations in the analysis."))
	}

    id_phase1 = c()
    for (var in X) {
        id_phase1 = union(id_phase1, which(is.na(data[,var])))
    }
	if (verbose) {
		print(paste("There are", n-length(id_phase1), "observations with complete measurements of expensive covariates."))
	}
    #### check data ###############################################################################################
	###############################################################################################################

	
	
	###############################################################################################################
	#### prepare analysis##########################################################################################	
    Y_vec = c(as.vector(data[-id_phase1,Y]), as.vector(data[id_phase1,Y]))
	storage.mode(Y_vec) = "double"
	
	if (!is.null(Delta))
	{
		Delta_vec = c(as.vector(data[-id_phase1,Delta]), as.vector(data[id_phase1,Delta]))
		storage.mode(Delta_vec) = "integer"
	}
	
	if (!is.null(L))
	{
	    L_vec = c(as.vector(data[-id_phase1,L]), as.vector(data[id_phase1,L]))
	    storage.mode(L_vec) = "double"
	}
	
    X_mat = as.matrix(data[-id_phase1,X])
	storage.mode(X_mat) = "double"
	
    if (!is.null(Z)) {
        Z_mat = rbind(as.matrix(data[-id_phase1,Z]), as.matrix(data[id_phase1,Z]))
		storage.mode(Z_mat) = "double"
        Bspline_Z_mat = rbind(as.matrix(data[-id_phase1,Bspline_Z]), as.matrix(data[id_phase1,Bspline_Z]))
		storage.mode(Bspline_Z_mat) = "double"
    }
	
	if (model == "coxph") {
		cov_names = X
	} else {
		cov_names = c("Intercept", X)
	}		
	if (!is.null(Z)) {
		cov_names = c(cov_names, Z)
	}
	
	ncov = length(cov_names)
	X_nc = length(X)
	rowmap = rep(NA, ncov)
	res_coefficients = matrix(NA, nrow=ncov, ncol=4)
	colnames(res_coefficients) = c("Estimate", "SE", "Statistic", "p-value")
	rownames(res_coefficients) = cov_names
	
	if (model %in% c("linear", "logistic")) {
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
	} else if (model == "coxph") {
	    if (is.null(Z)) {
	        Z_mat = NULL
	    }
		rowmap = 1:ncov
	}
	
	hn = hn_scale/sqrt(n)
	#### prepare analysis##########################################################################################
	###############################################################################################################
	

	
	###############################################################################################################
	#### analysis #################################################################################################
	if (model == "linear") {
		if (is.null(Z)) {
			res = .Call("TwoPhase_MLE0", Y_vec, X_mat, Z_mat, MAX_ITER, TOL, noSE, package="TwoPhaseReg")
		} else {
			res = .Call("TwoPhase_GeneralSpline", Y_vec, X_mat, Z_mat, Bspline_Z_mat, hn, MAX_ITER, TOL, noSE, package="TwoPhaseReg")
		}
	} else if (model == "logistic") {
		if (is.null(Z)) {
			res = .Call("TwoPhase_MLE0_logistic", Y_vec, X_mat, Z_mat, MAX_ITER, TOL, noSE, package="TwoPhaseReg")
		} else {
			res = .Call("TwoPhase_GeneralSpline_logistic", Y_vec, X_mat, Z_mat, Bspline_Z_mat, hn, MAX_ITER, TOL, noSE, package="TwoPhaseReg")
		}
	} else if (model == "coxph") {
		if (is.null(Z)) {
			if (is.null(W)) {
				res = .Call("TwoPhase_MLE0_noZW_coxph", Y_vec, Delta_vec, X_mat, MAX_ITER, TOL, noSE, package="TwoPhaseReg")
			} else {
				res = .Call("TwoPhase_MLE0_coxph", Y_vec, Delta_vec, X_mat, Z_mat, MAX_ITER, TOL, noSE, package="TwoPhaseReg")
			}
		} else {
		    if (is.null(L))
		    {
		        res = .Call("TwoPhase_GeneralSpline_coxph", Y_vec, Delta_vec, X_mat, Z_mat, Bspline_Z_mat, hn, MAX_ITER, TOL, noSE, package="TwoPhaseReg")
		    } else {
		        res = .Call("TwoPhase_GeneralSpline_coxph_LeftTrunc", Y_vec, L_vec, Delta_vec, X_mat, Z_mat, Bspline_Z_mat, hn, MAX_ITER, TOL, noSE, package="TwoPhaseReg")
		    }
			
		}
	}
    #### analysis #################################################################################################
	###############################################################################################################
	
	

    ###############################################################################################################
    #### return results ###########################################################################################
 	res_coefficients[,1] = res$theta[rowmap]
	res_coefficients[which(res_coefficients[,1] == -999),1] = NA
	res_coefficients[,2] = diag(res$cov_theta)[rowmap]
	res_coefficients[which(res_coefficients[,2] == -999),2] = NA
	res_coefficients[which(res_coefficients[,2] != -999),2] = sqrt(res_coefficients[which(res_coefficients[,2] != -999),2])
	
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
	res_final = list(coefficients=res_coefficients, convergence=!res$flag_nonconvergence, convergence_var=!res$flag_nonconvergence_cov)
	res_final
    #### return results ###########################################################################################
    ###############################################################################################################
}
