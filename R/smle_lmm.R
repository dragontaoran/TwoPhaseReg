#' Performs efficient semiparametric estimation for linear mixed models with longitudinal continuous outcomes under general two-phase designs.
#'
#' @param Y Specifies the column of the longitudinal continuous outcome. Observations with missing values of \code{Y} are omitted from the analysis. This argument is required.
#' @param Time Specifies the column of the time variable. Observations with missing values of \code{Time} are omitted from the analysis. This argument is required. 
#' @param ID Specifies the column of the subject ID. Subjects with missing values of \code{ID} are omitted from the analysis. This argument is required.
#' @param X Specifies the columns of the expensive covariates. \code{X} should be time-invariant. Subjects with missing values of \code{X} are considered as those not selected in the second phase. This argument is required.
#' @param Z Specifies the columns of the inexpensive covariates. \code{Z} should be time-invariant. Subjects with missing values of \code{Z} are omitted from the analysis. This argument is optional.
#' @param Bspline_Z Specifies the columns of the B-spline basis. Subjects with missing values of \code{Bspline_Z} are omitted from the analysis. This argument is not needed when \code{X} is independent of \code{Z}.
#' @param data Specifies the name of the dataset. This argument is required.
#' @param hn_scale Specifies the scale of the perturbation constant in the variance estimation. For example, if \code{hn_scale = 0.5}, then the perturbation constant is \eqn{0.5n^{-1/2}}, where \eqn{n} is the number of observations in the first-phase. The default value is \code{1}. This argument is optional. It is not needed when there is no \code{Z}.
#' @param MAX_ITER Specifies the maximum number of iterations in the EM algorithm. The default number is \code{2000}. This argument is optional.
#' @param TOL Specifies the convergence criterion in the EM algorithm. The default value is \code{1E-4}. This argument is optional.
#' @param noSE If \code{TRUE}, then the variances of the parameter estimators will not be estimated. The default value is \code{FALSE}. This argument is optional.
#' @param verbose If \code{TRUE}, then show details of the analysis. The default value is \code{FALSE}.
#' @param coef_initial Specifies the initial values of the fixed effects. \code{coef_initial} should be a numerical vector. The elements of \code{coef_initial} should follow the order of expensive covariates, intercept, inexpensive covariates (if any), time, and interactions between expensive covariates and time. This argument is optional. If this argument is not specified, then the maximum likelihood estimates from the null model without the expensive covariates will be used as initial values.
#' @param vc_initial Specifies the initial values of the variance component parameters. \code{vc_initial} should be a numerical vector of length \code{4}. The elements of \code{vc_initial} should follow the order of varaince of random intercept, covariance between random intercept and random slope of time, variance of random slope of time, and variance of the residual error term. This argument is optional. If this argument is not specified, then the maximum likelihood estimates from the null model without the expensive covariates will be used as initial values.
#' @return
#' \item{coefficients}{Stores the fixed effects estimates.}
#' \item{vc}{Stores the varaince component estimates.}
#' \item{converge}{In parameter estimation, if the EM algorithm converges, then \code{converge = TRUE}. Otherwise, \code{converge = FALSE}.}
#' \item{converge2}{In variance estimation, if the EM algorithm converges, then \code{converge2 = TRUE}. Otherwise, \code{converge2 = FALSE}.}
#' @useDynLib TwoPhaseReg
#' @importFrom Rcpp evalCpp
#' @importFrom stats pchisq
#' @exportPattern "^[[:alpha:]]+"
smle_lmm <- function (Y=NULL, Time=NULL, ID=NULL, X=NULL, Z=NULL, Bspline_Z=NULL, data=NULL, hn_scale=1, MAX_ITER=2000,
    TOL=1E-4, noSE=FALSE, verbose=FALSE, coef_initial=NULL, vc_initial=NULL) {

    ###############################################################################################################
    #### check data ###############################################################################################
    storage.mode(MAX_ITER) = "integer"
	storage.mode(TOL) = "double"
	storage.mode(noSE) = "integer"
	
	if (is.null(data)) {
	    stop("No dataset is provided!")
	}
	
	if (is.null(ID)) {
		stop("The subject ID is not specified!")
	}

	if (is.null(Y)) {
		stop("The response Y is not specified!")
	} else {
		vars_ph1 = Y
	}
	
	if (is.null(Time)) {
		stop("The observation time is not specified!")
	} else {
		vars_ph1 = c(vars_ph1, Time)
	}
	
	if (is.null(X)) {
		stop("The expensive covariates X are not specified!")
	}

	if (!is.null(Z)) {
		vars_ph1 = c(vars_ph1, Z)
	    if (!is.null(Bspline_Z)) {
    		vars_ph1 = c(vars_ph1, Bspline_Z)
	    }
	} else {
		if (!is.null(Bspline_Z)) {
			stop("Cannot specify Bspline_Z when Z is not specified!")
		}
	}
	
    id_exclude = c()
    for (var in vars_ph1) {
        id_exclude = union(id_exclude, which(is.na(data[,var])))
    }
	
	if (verbose) {
    	print(paste("There are", nrow(data), "observations in the dataset."))
    	print(paste(length(id_exclude), "observations are excluded due to missing Y, Time, Z, or Bspline_Z."))
	}
	if (length(id_exclude) > 0) {
		data = data[-id_exclude,]
	}
	
    n = nrow(data)
	if (verbose) {
    	print(paste("There are", n, "observations in the analysis."))
	}
	
	### sort data by ID and time
	data = data[order(data[,ID], data[,Time]),]
	
	### detect observations for subjects not selected in the second phase
    obs_phase1_tmp = c()
    for (var in X) {
        obs_phase1_tmp = union(obs_phase1_tmp, which(is.na(data[,var])))
    }
	if (verbose) {
		print(paste("There are", n-length(obs_phase1_tmp), "observations with complete measurements of expensive covariates."))
	}
	
	### check data by subject
	ids = unique(data[,ID])
	nid = length(ids)
	if (verbose) {
		print(paste("There are", nid, "subjects in the analysis."))
	}	
	nid2 = length(unique(data[-obs_phase1_tmp,ID]))
	if (verbose) {
		print(paste("There are", nid2, "subjects with complete measurements of expensive covariates."))
	}
	
	obs_phase1 = c()
	obs_phase2 = c()
	id_phase1 = c()
	id_phase2 = c()
	index_obs_phase1 = c()
	index_obs_phase2 = c()
	ind_phase1 = 0
	ind_phase2 = 0
	for (i in 1:nid) {
		obs = which(data[,ID] == ids[i])
		nobs = length(obs)
		if (nobs > 1) {
			for (j in 1:(nobs-1)) {
				if (data[obs[j],Time] == data[obs[j+1],Time]) {
					stop(paste0("Two observations with the same times detected for subject ", ids[i], "!"))
				}
				if (!is.null(Z)) {
					if (any(as.vector(data[obs[j],Z]) != as.vector(data[obs[j+1],Z]))) {
						stop(paste0("Two observations with different Z detected for subject ", ids[i], "!"))
					}
					if (!is.null(Bspline_Z)) {
						if (any(as.vector(data[obs[j],Bspline_Z]) != as.vector(data[obs[j+1],Bspline_Z]))) {
							stop(paste0("Two observations with different Bspline_Z detected for subject ", ids[i], "!"))
						}
					}
				}
			}
			nobs_phase1 = sum(obs %in% obs_phase1_tmp)
			if (nobs_phase1 == 0) {
				for (j in 1:(nobs-1)) {
					if (any(as.vector(data[obs[j],X]) != as.vector(data[obs[j+1],X]))) {
						stop(paste0("Two observations with different X detected for subject ", ids[i], "!"))
					}
				}
				obs_phase2 = c(obs_phase2, obs)
				id_phase2 = c(id_phase2, obs[1])
				index_obs_phase2 = c(index_obs_phase2, ind_phase2, nobs)
				ind_phase2 = ind_phase2+nobs
			} else if (nobs_phase1 < nobs) {
				stop(paste0("Two observations with different X detected for subject ", ids[i], "!"))
			} else {
				obs_phase1 = c(obs_phase1, obs)
				id_phase1 = c(id_phase1, obs[1])
				index_obs_phase1 = c(index_obs_phase1, ind_phase1, nobs)
				ind_phase1 = ind_phase1+nobs				
			}
		} else {
			if (obs %in% obs_phase1_tmp) {
				obs_phase1 = c(obs_phase1, obs)
				id_phase1 = c(id_phase1, obs[1])
				index_obs_phase1 = c(index_obs_phase1, ind_phase1, nobs)
				ind_phase1 = ind_phase1+nobs
			} else {
				obs_phase2 = c(obs_phase2, obs)
				id_phase2 = c(id_phase2, obs[1])
				index_obs_phase2 = c(index_obs_phase2, ind_phase2, nobs)
				ind_phase2 = ind_phase2+nobs				
			}
		}
	}
	
	if (length(obs_phase1)+length(obs_phase2) != n) {
		stop("length(obs_phase1)+length(obs_phase2) != n!")
	}
	if (length(id_phase1)+length(id_phase2) != nid) {
		stop("length(id_phase1)+length(id_phase2) != nid!")
	}
    #### check data ###############################################################################################
	###############################################################################################################

	
	
	###############################################################################################################
	#### prepare analysis #########################################################################################
    Y_vec = c(as.vector(data[obs_phase2,Y]), as.vector(data[obs_phase1,Y]))
	storage.mode(Y_vec) = "double"
	
	T_vec = c(as.vector(data[obs_phase2,Time]), as.vector(data[obs_phase1,Time]))
	storage.mode(T_vec) = "double"
			
    X_mat = as.matrix(data[id_phase2,X])
	storage.mode(X_mat) = "double"
	
    if (!is.null(Z)) {
        Z_mat = rbind(as.matrix(data[id_phase2,Z]), as.matrix(data[id_phase1,Z]))
		storage.mode(Z_mat) = "double"
		if (!is.null(Bspline_Z)) {
			Bspline_Z_mat = rbind(as.matrix(data[id_phase2,Bspline_Z]), as.matrix(data[id_phase1,Bspline_Z]))
			storage.mode(Bspline_Z_mat) = "double"
		}
    }
	
	index_obs = rbind(matrix(index_obs_phase2, ncol=2, byrow=TRUE), matrix(index_obs_phase1, ncol=2, byrow=TRUE))
	if (nrow(index_obs) != nid) {
		stop("ncol(index_obs) != nid!")
	}
	index_obs[-(1:length(id_phase2)),1] = index_obs[-(1:length(id_phase2)),1]+length(obs_phase2)
	storage.mode(index_obs) = "integer"
	
	cov_names = c("Intercept", X)		
	if (!is.null(Z)) {
		cov_names = c(cov_names, Z)
	}
	cov_names = c(cov_names, "Time", paste0("Time_", X))
	
	ncov = length(cov_names)
	X_nc = length(X)

	if (is.null(Z)) {
		Z_mat = rep(1., nid)
		Z_nc = 1;
	} else {
		Z_mat = cbind(1, Z_mat)
		Z_nc = ncol(Z_mat)
	}
	
	hn = hn_scale/sqrt(n)
	#### prepare analysis #########################################################################################
	###############################################################################################################
	

	
	###############################################################################################################
	#### analysis #################################################################################################
	require(lme4)

	if (is.null(Bspline_Z)) {
	} else {
	    if (is.null(coef_initial) | is.null(vc_initial)) {
	        res0_formula = as.formula(paste0(Y, "~", paste(Z, collapse="+"), "+", Time, "+(", Time, "|", ID, ")"))
	        res0 = summary(lmer(formula=res0_formula, data=data, REML=FALSE))
	        coef_initial = rep(NA, ncov)
	        coef_initial[1:X_nc] = 0;
	        coef_initial[(X_nc+1):(X_nc+Z_nc+1)] = coef(res0)[,1]
	        coef_initial[(X_nc+Z_nc+2):(X_nc+Z_nc+1+X_nc)] = 0
	        vc_initial = rep(NA, 4)
	        vc_initial[1] = res0$varcor[[1]][1,1]
	        vc_initial[2] = res0$varcor[[1]][1,2]
	        vc_initial[3] = res0$varcor[[1]][2,2]
	        vc_initial[4] = res0$sigma^2
	    }
		res = .Call("LMM_GeneralSpline", Y_vec, T_vec, X_mat, Z_mat, Bspline_Z_mat, index_obs, coef_initial, vc_initial, hn, MAX_ITER, TOL, noSE, package="TwoPhaseReg")
	}
    #### analysis #################################################################################################
	###############################################################################################################
	
	

    ###############################################################################################################
    #### return results ###########################################################################################
	
	res_coefficients = matrix(NA, nrow=ncov, ncol=4)
	colnames(res_coefficients) = c("Estimate", "SE", "Statistic", "p-value")
	rownames(res_coefficients) = cov_names
	
	rowmap = rep(NA, ncov)
	rowmap[1] = X_nc+1
	rowmap[2:(X_nc+1)] = 1:X_nc
	rowmap[(X_nc+2):ncov] = (X_nc+2):ncov
	
	res_vc = rep(NA, 4)
	names(res_vc) = c("Intercept","Covariance","Slope","Residual Variance")
	
 	res_coefficients[,1] = res$theta[rowmap]
	res_coefficients[which(res_coefficients[,1] == -999),1] = NA
	res_coefficients[,2] = diag(res$cov_theta)[rowmap]
	res_coefficients[which(res_coefficients[,2] <= 0),2] = NA
	res_coefficients[which(res_coefficients[,2] > 0),2] = sqrt(res_coefficients[which(res_coefficients[,2] > 0),2])
	res_vc = res$vcparams
	res_vc[which(res_vc == -999)] = NA
	
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
	res_final = list(coefficients=res_coefficients, vc=res_vc, converge=!res$flag_nonconvergence, converge2=!res$flag_nonconvergence_cov)
	res_final
    #### return results ###########################################################################################
    ###############################################################################################################
}
