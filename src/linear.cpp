#include <RcppEigen.h>
#include <iostream>
#include <ctime>
#include <cmath>
#include <string>
#include "utility.h"

using namespace Rcpp;

typedef Map<VectorXd> MapVecd;
typedef Map<VectorXi> MapVeci;
typedef Map<MatrixXd> MapMatd;

double WaldLinearGeneralSplineProfile (Ref<MatrixXd> pB, Ref<RowVectorXd> p_col_sum,
	Ref<VectorXd> q_row_sum, Ref<MatrixXd> p, Ref<MatrixXd> p0, Ref<MatrixXd> P_theta, Ref<MatrixXd> q, Ref<VectorXd> resi_n, Ref<MatrixXd> logp,
	const Ref<const VectorXd>& theta, const Ref<const VectorXd>& Y, const Ref<const MatrixXd>& X, const Ref<const MatrixXd>& Bspline_uni,
	const Ref<const MatrixXd>& ZW, const Ref<const MatrixXd>& X_uni, const Ref<const VectorXi>& X_uni_ind, 
	const Ref<const VectorXi>& Bspline_uni_ind, const Ref<const MatrixXd>& p_static, const double sigma_sq, const int n, const int n2, const int m, 
	const int s, const int n_minus_n2, const int X_nc, const int ZW_nc, const int MAX_ITER, const double TOL) 
{
	/**** temporary variables **********************************************************************************************************************/	
	double tol;
	int iter;
	/* test code */
	// time_t t1, t2;
	/* test code end */
	/**** temporary variables **********************************************************************************************************************/
	
	/**** update P_theta ***************************************************************************************************************************/
	P_theta.col(0) = Y.tail(n_minus_n2);
	P_theta.col(0).noalias() -= ZW.bottomRows(n_minus_n2)*theta.tail(ZW_nc);
	for (int k=1; k<m; k++) 
	{
		P_theta.col(k) = P_theta.col(0);
	}		
	for (int k=0; k<m; k++) 
	{
		P_theta.col(k).noalias() -= VectorXd::Constant(n_minus_n2, (X_uni.row(k)*theta.head(X_nc))(0,0));	
	}		
	P_theta = P_theta.array().square();
	P_theta /= -2.*sigma_sq;
	P_theta = P_theta.array().exp();
	/**** update P_theta ***************************************************************************************************************************/
		
	/**** parameter initialization *****************************************************************************************************************/
	p_col_sum = p_static.colwise().sum();
	for (int j=0; j<s; j++) 
	{
		p.col(j) = p_static.col(j)/p_col_sum(j);
	}
	p0 = p;
	/**** parameter initialization *****************************************************************************************************************/
	
	for (iter=0; iter<MAX_ITER; iter++) 
	{
		/* test code */
		// time(&t1);
		/* test code end */
		
		/**** E-step *******************************************************************************************************************************/
		
		/**** update pB ****************************************************************************************************************************/
		pB = Bspline_uni*p.transpose();
		/**** update pB ****************************************************************************************************************************/
				
		/**** update q, q_row_sum ******************************************************************************************************************/
		for (int i=0; i<n_minus_n2; i++) 
		{
			for (int k=0; k<m; k++) 
			{
				q(i,k) = P_theta(i,k)*pB(Bspline_uni_ind(i+n2),k);
			}
		}
		
		q_row_sum = q.rowwise().sum();		
		for (int i=0; i<n_minus_n2; i++) 
		{
			q.row(i) /= q_row_sum(i);
		}
		/**** update q, q_row_sum ******************************************************************************************************************/
		
		/**** E-step *******************************************************************************************************************************/
			
		
		/**** M-step *******************************************************************************************************************************/
		
		/**** update p *****************************************************************************************************************************/
		p.setZero();		
		for (int i=0; i<n_minus_n2; i++) 
		{
			for (int k=0; k<m; k++) 
			{
				for (int j=0; j<s; j++) 
				{
					p(k,j) += Bspline_uni(Bspline_uni_ind(i+n2),j)*P_theta(i,k)/q_row_sum(i);
				}
			}
		}		
		p = p.array()*p0.array();
		p += p_static;
		p_col_sum = p.colwise().sum();
		for (int j=0; j<s; j++) 
		{
			p.col(j) /= p_col_sum(j);
		}
		/**** update p *****************************************************************************************************************************/
		
		/**** M-step *******************************************************************************************************************************/
				
		/**** calculate the sum of absolute differences between estimates in the current and previous iterations ***********************************/
		tol = (p-p0).array().abs().sum();
		/**** calculate the sum of absolute differences between estimates in the current and previous iterations ***********************************/
								
		/**** update parameters ********************************************************************************************************************/
		p0 = p;
		/**** update parameters ********************************************************************************************************************/
		
		/**** check convergence ********************************************************************************************************************/
		if (tol < TOL) 
		{
			break;
		}
		/**** check convergence ********************************************************************************************************************/
		
		/* test code */
		// time(&t2);
		// Rcpp::Rcout << iter << '\t' << difftime(t2, t1) << '\t' << tol << endl;
		/* test code end */
	}
	
	if (iter == MAX_ITER) 
	{
		return -999.;
	} 
	else 
	{
		/**** calculate the likelihood *************************************************************************************************************/
		double tmp, loglik;
		
		logp = p.array().log();
		for (int k=0; k<m; k++)
		{
			for (int j=0; j<s; j++)
			{
				if (p(k,j) <= 0.)
				{
					logp(k,j) = 0.;
				}
			}
		}
		pB = Bspline_uni*logp.transpose();
		
		loglik = 0.;
		for (int i=0; i<n2; i++) {
			loglik += pB(Bspline_uni_ind(i),X_uni_ind(i));
		}
		
		pB = Bspline_uni*p.transpose();
		
		for (int i=0; i<n_minus_n2; i++) 
		{
			for (int k=0; k<m; k++) 
			{
				q(i,k) = P_theta(i,k)*pB(Bspline_uni_ind(i+n2),k);
			}
		}
		q_row_sum = q.rowwise().sum();		
		
		loglik += q_row_sum.array().log().sum();		
		loglik += -log(2.*M_PI*sigma_sq)*n/2.;
		
		resi_n = Y.head(n2);
		resi_n.noalias() -= X*theta.head(X_nc);
		resi_n.noalias() -= ZW.topRows(n2)*theta.tail(ZW_nc);
		
		tmp = resi_n.squaredNorm();
		tmp /= 2.*sigma_sq;
		loglik -= tmp;
		/**** calculate the likelihood *************************************************************************************************************/
		
		return loglik;	
	}
} // WaldLinearGeneralSplineProfile

RcppExport SEXP TwoPhase_GeneralSpline (SEXP Y_R, SEXP X_R, SEXP ZW_R, SEXP Bspline_R, SEXP hn_R, SEXP MAX_ITER_R, SEXP TOL_R, SEXP noSE_R) 
{
	/*#############################################################################################################################################*/
	/**** pass arguments from R to cpp *************************************************************************************************************/
	const MapVecd Y(as<MapVecd>(Y_R));
	const MapMatd X(as<MapMatd>(X_R));
	const MapMatd ZW(as<MapMatd>(ZW_R));
	const MapMatd Bspline(as<MapMatd>(Bspline_R));
	const double hn = NumericVector(hn_R)[0];
	const int MAX_ITER = IntegerVector(MAX_ITER_R)[0];
	const double TOL = NumericVector(TOL_R)[0];
	const int noSE = IntegerVector(noSE_R)[0];
	/**** pass arguments from R to cpp *************************************************************************************************************/
	/*#############################################################################################################################################*/
	
	
	
	/*#############################################################################################################################################*/
	/**** some useful constants ********************************************************************************************************************/
	const int n = Y.size();  // number of subjects in the first phase
	const int n2 = X.rows(); // number of subjects in the second phase
	const int n_minus_n2 = n-n2; // number of subjects not selected in the second phase
	const int ZW_nc = ZW.cols(); // number of inexpensive covariates
	const int X_nc = X.cols(); // number of expensive covariates X
	const int ncov = X_nc+ZW_nc; // number of all covariates
	const int s = Bspline.cols(); // number of B-spline functions
	/**** some useful constants ********************************************************************************************************************/
	/*#############################################################################################################################################*/	
	

	
	/*#############################################################################################################################################*/	
	/**** summarize observed distinct rows of X ****************************************************************************************************/
	// m: number of distinct rows of X
	// X_uni_ind(n2): row index of rows of X in X_uni
	// X_uni(m, X_nc): distinct rows of X
	// X_uni_n(m): count of appearances of each distinct row of X	
	int m;
	VectorXi X_index(n2);
	VectorXi X_uni_ind(n2);	
	indexx_Matrix_Row(X, X_index);
	Num_Uni_Matrix_Row(X, X_index, m);	
	MatrixXd X_uni(m, X_nc); 
	VectorXi X_uni_n(m);	
	Create_Uni_Matrix_Row(X, X_index, X_uni, X_uni_ind, X_uni_n);
	/**** summarize observed distinct rows of X ****************************************************************************************************/
	/*#############################################################################################################################################*/
	

	
	/*#############################################################################################################################################*/
	/**** summarize observed distinct rows of Bspline **********************************************************************************************/
	// m_B: number of distinct rows of Bspline
	// Bspline_uni_ind(n): row index of rows of Bspline in Bspline_uni
	// Bspline_uni(m_B, s): distinct rows of Bspline
	// Bspline_uni_n(m_B): count of appearances of each distinct row of Bspline
	int m_B;
	VectorXi Bspline_index(n);
	VectorXi Bspline_uni_ind(n);	
	indexx_Matrix_Row(Bspline, Bspline_index);
	Num_Uni_Matrix_Row(Bspline, Bspline_index, m_B);	
	MatrixXd Bspline_uni(m_B, s);
	VectorXi Bspline_uni_n(m_B);
	Create_Uni_Matrix_Row(Bspline, Bspline_index, Bspline_uni, Bspline_uni_ind, Bspline_uni_n);
	/**** summarize observed distinct rows of Bspline **********************************************************************************************/
	/*#############################################################################################################################################*/
	
	
	
	/*#############################################################################################################################################*/
	/**** some fixed quantities in the EM algorithm ************************************************************************************************/	
	// t(Y)*Y
	const double LS_YtY_static = Y.squaredNorm(); 
	
	// t(X,ZW)*Y
	VectorXd LS_XtY_static(ncov); 
	LS_XtY_static.head(X_nc) = X.transpose()*Y.head(n2);
	LS_XtY_static.tail(ZW_nc) = ZW.transpose()*Y;
	
	// t(X,ZW)*(X,ZW)
	MatrixXd LS_XtX_static(ncov, ncov); 
	LS_XtX_static.topLeftCorner(X_nc,X_nc) = X.transpose()*X;
	LS_XtX_static.topRightCorner(X_nc,ZW_nc) = X.transpose()*ZW.topRows(n2);
	LS_XtX_static.bottomRightCorner(ZW_nc,ZW_nc) = ZW.transpose()*ZW;
	
	// p
	MatrixXd p_static(m, s); 
	p_static.setZero();
	for (int i=0; i<n2; i++) 
	{
		p_static.row(X_uni_ind(i)) += Bspline.row(i);
	}	
	/**** some fixed quantities in the EM algorithm ************************************************************************************************/
	/*#############################################################################################################################################*/
	

	
	/*#############################################################################################################################################*/
	/**** output ***********************************************************************************************************************************/
	VectorXd theta(ncov); // regression coefficients
	MatrixXd cov_theta(ncov, ncov); // covariance matrix
	bool flag_nonconvergence; // flag of none convergence in the estimation of regression coefficients
	bool flag_nonconvergence_cov; // flag of none convergence in the estimation of covariance matrix
	double sigma_sq; // residual variance
	/**** output ***********************************************************************************************************************************/
	/*#############################################################################################################################################*/
	

		
	/*#############################################################################################################################################*/
	/**** temporary variables **********************************************************************************************************************/
	VectorXd LS_XtY(ncov);
	MatrixXd LS_XtX(ncov, ncov);
	VectorXd theta0(ncov);
	double sigma_sq0;
	MatrixXd p(m, s); 
	MatrixXd p0(m, s);
	RowVectorXd p_col_sum(s);	
	MatrixXd q(n_minus_n2, m);
	VectorXd q_row_sum(n_minus_n2);
	RowVectorXd q_col_sum(m);
	MatrixXd pB(m_B, m);
	MatrixXd P_theta(n_minus_n2, m);
	double tol;
	int iter, idx;
	// /* RT's test code */
	// time_t t1, t2;
	// /* RT's test code end */
	/**** temporary variables **********************************************************************************************************************/
	/*#############################################################################################################################################*/
	

	
	/*#############################################################################################################################################*/
	/**** EM algorithm *****************************************************************************************************************************/
	
	/**** parameter initialization *****************************************************************************************************************/
	theta.setZero();
	theta0.setZero();
	sigma_sq = sigma_sq0 = Var(Y);

	p_col_sum = p_static.colwise().sum();
	for (int j=0; j<s; j++) 
	{
		p.col(j) = p_static.col(j)/p_col_sum(j);
	}
	p0 = p;
	
	flag_nonconvergence = false;
	flag_nonconvergence_cov = false;
	/**** parameter initialization *****************************************************************************************************************/
	
	for (iter=0; iter<MAX_ITER; iter++) 
	{
		// /* RT's test code */
		// time(&t1);
		// /* RT's test code end */
		
		/**** E-step *******************************************************************************************************************************/
		
		/**** update pB ****************************************************************************************************************************/
		pB = Bspline_uni*p.transpose();
		/**** update pB ****************************************************************************************************************************/
				
		/**** update P_theta ***********************************************************************************************************************/
		P_theta.col(0) = Y.tail(n_minus_n2);
		P_theta.col(0).noalias() -= ZW.bottomRows(n_minus_n2)*theta.tail(ZW_nc);		
		for (int k=1; k<m; k++) 
		{
			P_theta.col(k) = P_theta.col(0);
		}		
		for (int k=0; k<m; k++) 
		{
			P_theta.col(k).noalias() -= VectorXd::Constant(n_minus_n2, (X_uni.row(k)*theta.head(X_nc))(0,0));	
		}		
		P_theta = P_theta.array().square();
		P_theta /= -2.*sigma_sq;
		P_theta = P_theta.array().exp();
		/**** update P_theta ***********************************************************************************************************************/
		
		/**** update q, q_row_sum ******************************************************************************************************************/
		for (int i=0; i<n_minus_n2; i++) 
		{
			for (int k=0; k<m; k++) 
			{
				q(i,k) = P_theta(i,k)*pB(Bspline_uni_ind(i+n2),k);
			}
		}
		q_row_sum = q.rowwise().sum();		
		for (int i=0; i<n_minus_n2; i++) 
		{
			q.row(i) /= q_row_sum(i);
		}
		q_col_sum = q.colwise().sum();
		/**** update q, q_row_sum ******************************************************************************************************************/
		
		/**** E-step *******************************************************************************************************************************/
		
		
		/**** M-step *******************************************************************************************************************************/
		
		/**** update theta and sigma_sq ************************************************************************************************************/		
		LS_XtX = LS_XtX_static;
		LS_XtY = LS_XtY_static;
		
		for (int k=0; k<m; k++) 
		{
			LS_XtX.topLeftCorner(X_nc,X_nc).noalias() += q_col_sum(k)*X_uni.row(k).transpose()*X_uni.row(k);
		}
		
		for (int i=0; i<n_minus_n2; i++) 
		{
			idx = i+n2;
			for (int k=0; k<m; k++) 
			{
				LS_XtX.topRightCorner(X_nc,ZW_nc).noalias() += q(i,k)*X_uni.row(k).transpose()*ZW.row(idx);
				LS_XtY.head(X_nc).noalias() += q(i,k)*X_uni.row(k).transpose()*Y(idx);				
			}
		}
		
		theta = LS_XtX.selfadjointView<Eigen::Upper>().ldlt().solve(LS_XtY);		
		sigma_sq = LS_XtY.transpose()*theta;
		sigma_sq = (LS_YtY_static-sigma_sq)/n;
		/**** update theta and sigma_sq ************************************************************************************************************/
		
		/**** update p *****************************************************************************************************************************/
		p.setZero();		
		for (int i=0; i<n_minus_n2; i++) 
		{
			for (int k=0; k<m; k++) 
			{
				for (int j=0; j<s; j++) 
				{
					p(k,j) += Bspline_uni(Bspline_uni_ind(i+n2),j)*P_theta(i,k)/q_row_sum(i);
				}
			}
		}		
		p = p.array()*p0.array();
		p += p_static;
		p_col_sum = p.colwise().sum();
		for (int j=0; j<s; j++) 
		{
			p.col(j) /= p_col_sum(j);
		}
		/**** update p *****************************************************************************************************************************/
		
		/**** M-step *******************************************************************************************************************************/
		
		
		/**** calculate the sum of absolute differences between estimates in the current and previous iterations ***********************************/
		tol = (theta-theta0).array().abs().sum();
		tol += fabs(sigma_sq-sigma_sq0);
		tol += (p-p0).array().abs().sum();
		/**** calculate the sum of absolute differences between estimates in the current and previous iterations ***********************************/		
				
		/**** update parameters ********************************************************************************************************************/
		theta0 = theta;
		sigma_sq0 = sigma_sq;
		p0 = p;
		/**** update parameters ********************************************************************************************************************/
		
		/**** check convergence ********************************************************************************************************************/
		if (tol < TOL) 
		{
			break;
		}
		/**** check convergence ********************************************************************************************************************/
		
		// /* RT's test code */
		// time(&t2);
		// Rcout << iter << '\t' << difftime(t2, t1) << '\t' << tol << endl;
		// /* RT's test code end */
	}
	/**** EM algorithm *****************************************************************************************************************************/
	/*#############################################################################################################################################*/
	
	
	
	/*#############################################################################################################################################*/
	/**** variance estimation **********************************************************************************************************************/
	if (iter == MAX_ITER) 
	{
		flag_nonconvergence = true;
		flag_nonconvergence_cov = true;
		theta.setConstant(-999.);
		sigma_sq = -999.;
		cov_theta.setConstant(-999.);
	} 
	else if (noSE) 
	{
		flag_nonconvergence_cov = true;
		cov_theta.setConstant(-999.);
	} 
	else 
	{
		VectorXd resi_n(n2);
		VectorXd profile_vec(ncov+1);
		MatrixXd logp(m, s);
		MatrixXd profile_mat(ncov+1, ncov+1);
		MatrixXd inv_profile_mat(ncov+1, ncov+1);
		double loglik;
		
		profile_mat.setZero();
		profile_vec.setZero();
		
		loglik = WaldLinearGeneralSplineProfile(pB, p_col_sum, q_row_sum, p, p0, P_theta, q, resi_n, logp, theta, Y, X, Bspline_uni, ZW, X_uni,
			X_uni_ind, Bspline_uni_ind, p_static, sigma_sq, n, n2, m, s, n_minus_n2, X_nc, ZW_nc, MAX_ITER, TOL);
		if (loglik == -999.) 
		{
			flag_nonconvergence_cov = true;
		}
		for (int i=0; i<ncov+1; i++) 
		{
			for (int j=i; j<ncov+1; j++) 
			{
				profile_mat(i,j) = loglik;
			}
		}
		 		
		for (int i=0; i<ncov; i++) 
		{
			theta0 = theta;
			theta0(i) += hn;
			sigma_sq0 = sigma_sq;
			profile_vec(i) = WaldLinearGeneralSplineProfile(pB, p_col_sum, q_row_sum, p, p0, P_theta, q, resi_n, logp, theta0, Y, X, Bspline_uni, ZW, X_uni,
				X_uni_ind, Bspline_uni_ind, p_static, sigma_sq0, n, n2, m, s, n_minus_n2, X_nc, ZW_nc, MAX_ITER, TOL);
		}
		theta0 = theta;
		sigma_sq0 = sigma_sq+hn;	
		profile_vec(ncov) = WaldLinearGeneralSplineProfile(pB, p_col_sum, q_row_sum, p, p0, P_theta, q, resi_n, logp, theta0, Y, X, Bspline_uni, ZW, X_uni,
			X_uni_ind, Bspline_uni_ind, p_static, sigma_sq0, n, n2, m, s, n_minus_n2, X_nc, ZW_nc, MAX_ITER, TOL);
		for (int i=0; i<ncov+1; i++) 
		{
			if(profile_vec(i) == -999.) 
			{
				flag_nonconvergence_cov = true;
			}
		}
		
		for (int i=0; i<ncov; i++) 
		{
			theta0 = theta;
			theta0(i) += hn;
			sigma_sq0 = sigma_sq+hn;
			loglik = WaldLinearGeneralSplineProfile(pB, p_col_sum, q_row_sum, p, p0, P_theta, q, resi_n, logp, theta0, Y, X, Bspline_uni, ZW, X_uni,
				X_uni_ind, Bspline_uni_ind, p_static, sigma_sq0, n, n2, m, s, n_minus_n2, X_nc, ZW_nc, MAX_ITER, TOL);
			if (loglik == -999.) 
			{
				flag_nonconvergence_cov = true;
			}
			profile_mat(i,ncov) += loglik;
			profile_mat(i,ncov) -= profile_vec(i)+profile_vec(ncov);
			for (int j=i; j<ncov; j++) 
			{
				theta0 = theta;
				theta0(i) += hn;
				theta0(j) += hn;
				sigma_sq0 = sigma_sq;
				loglik = WaldLinearGeneralSplineProfile(pB, p_col_sum, q_row_sum, p, p0, P_theta, q, resi_n, logp, theta0, Y, X, Bspline_uni, ZW, X_uni,
					X_uni_ind, Bspline_uni_ind, p_static, sigma_sq0, n, n2, m, s, n_minus_n2, X_nc, ZW_nc, MAX_ITER, TOL);
				if (loglik == -999.) 
				{
					flag_nonconvergence_cov = true;
				}				
				profile_mat(i,j) += loglik;
				profile_mat(i,j) -= profile_vec(i)+profile_vec(j);
			}
		}
		theta0 = theta;
		sigma_sq0 = sigma_sq+2.*hn;
		loglik = WaldLinearGeneralSplineProfile(pB, p_col_sum, q_row_sum, p, p0, P_theta, q, resi_n, logp, theta0, Y, X, Bspline_uni, ZW, X_uni,
			X_uni_ind, Bspline_uni_ind, p_static, sigma_sq0, n, n2, m, s, n_minus_n2, X_nc, ZW_nc, MAX_ITER, TOL);
		if (loglik == -999.) 
		{
			flag_nonconvergence_cov = true;
		}
		profile_mat(ncov,ncov) += loglik;
		profile_mat(ncov,ncov) -= 2.*profile_vec(ncov);
				
		if (flag_nonconvergence_cov == true) 
		{
			cov_theta.setConstant(-999.);
		} 
		else 
		{		
			for (int i=0; i<ncov+1; i++) 
			{
				for (int j=i+1; j<ncov+1; j++) 
				{
					profile_mat(j,i) = profile_mat(i,j);
				}
			}		
			profile_mat /= hn*hn;
			profile_mat = -profile_mat;
			inv_profile_mat = profile_mat.selfadjointView<Eigen::Upper>().ldlt().solve(MatrixXd::Identity(ncov+1, ncov+1));
			cov_theta = inv_profile_mat.topLeftCorner(ncov,ncov);
		}
	}
	/**** variance estimation **********************************************************************************************************************/
	/*#############################################################################################################################################*/
	
	
	
	/*#############################################################################################################################################*/
	/**** return output to R ***********************************************************************************************************************/
	return List::create(Named("theta") = theta,
						Named("sigma_sq") = sigma_sq,
						Named("cov_theta") = cov_theta,
						Named("flag_nonconvergence") = flag_nonconvergence,
						Named("flag_nonconvergence_cov") = flag_nonconvergence_cov);
	/**** return output to R ***********************************************************************************************************************/
	/*#############################################################################################################################################*/
} // TwoPhase_GeneralSpline

void WaldLinearVarianceMLE0 (Ref<MatrixXd> cov_theta, const Ref<const MatrixXd>& LS_XtX, const Ref<const VectorXd>& LS_XtY, const Ref<const VectorXd>& p,
	const Ref<const VectorXd>& Y, const Ref<const MatrixXd>& ZW, const Ref<const VectorXd>& theta, const Ref<const MatrixXd>& X_uni, const Ref<const MatrixXd>& q, 
	const int n_minus_n2, const int X_nc, const int ZW_nc, const int m, const int ncov, const int n, const int n2, const double sigma_sq) 
{
	const int dimQ = ncov+1+m;

	VectorXd l1i(dimQ);
	VectorXd l1ik(dimQ);
	MatrixXd Q(dimQ, dimQ);
	MatrixXd resi(n_minus_n2, m);
	MatrixXd inv_profile_mat(dimQ-1, dimQ-1);
	
	/**** calculate resi ***************************************************************************************************************************/
	resi.col(0) = Y.tail(n_minus_n2);
	resi.col(0).noalias() -= ZW.bottomRows(n_minus_n2)*theta.tail(ZW_nc);

	for (int k=1; k<m; k++) {
		resi.col(k) = resi.col(0);
	}

	for (int k=0; k<m; k++) {
		resi.col(k).noalias() -= (X_uni.row(k)*theta.head(X_nc)).replicate(n_minus_n2,1);
	}
	/**** calculate resi ***************************************************************************************************************************/
	

	
	/**** augment the upper diagonal of Q **********************************************************************************************************/	
	// add l2 
	Q.setZero();
	Q.topLeftCorner(ncov,ncov) = LS_XtX/sigma_sq;
	Q.block(0,ncov,ncov,1) = (LS_XtY-LS_XtX*theta)/(sigma_sq*sigma_sq);
	Q(ncov,ncov) = (n+0.)/(2*sigma_sq*sigma_sq);
	for (int k=0; k<m; k++) {
		Q(ncov+1+k,ncov+1+k) = (n+0.)/p(k);
	}
	
	// add l1i, l1ik
	for (int i=0; i<n_minus_n2; i++) {
		l1i.setZero();
		for (int k=0; k<m; k++) {
			l1ik.setZero();
			l1ik.head(X_nc) = resi(i,k)*(X_uni.row(k).transpose())/sigma_sq;
			l1ik.segment(X_nc,ZW_nc) = resi(i,k)*(ZW.row(i+n2).transpose())/sigma_sq;
			l1ik(ncov) = -1./(2*sigma_sq)+resi(i,k)*resi(i,k)/(2*sigma_sq*sigma_sq);
			l1ik(ncov+1+k) = 1./p(k);
			l1i += q(i,k)*l1ik;
			Q -= q(i,k)*l1ik*l1ik.transpose();
		}
		Q += l1i*l1i.transpose();
	}
		
	for (int k=0; k<m-1; k++) {
		Q.block(0,ncov+1+k,ncov+1,1) -= Q.topRightCorner(ncov+1,1);
		for (int kk=k; kk<m-1; kk++) {
			Q(ncov+1+k,ncov+1+kk) -= Q(ncov+1+k,dimQ-1)+Q(ncov+1+kk,dimQ-1)-Q(dimQ-1,dimQ-1);
		}
	}
	
	/**** augment the upper diagonal of Q **********************************************************************************************************/		
	inv_profile_mat = Q.topLeftCorner(dimQ-1,dimQ-1).selfadjointView<Eigen::Upper>().ldlt().solve(MatrixXd::Identity(dimQ-1, dimQ-1));
	cov_theta = inv_profile_mat.topLeftCorner(ncov,ncov);	
} // WaldLinearVarianceMLE0

RcppExport SEXP TwoPhase_MLE0 (SEXP Y_R, SEXP X_R, SEXP ZW_R, SEXP MAX_ITER_R, SEXP TOL_R, SEXP noSE_R)
/**** when there is no Z ***************************************************************************************************************************/
{
	/*#############################################################################################################################################*/
	/**** pass arguments from R to cpp *************************************************************************************************************/
	const MapVecd Y(as<MapVecd>(Y_R));
	const MapMatd X(as<MapMatd>(X_R));
	const MapMatd ZW(as<MapMatd>(ZW_R));
	const int MAX_ITER = IntegerVector(MAX_ITER_R)[0];
	const double TOL = NumericVector(TOL_R)[0];
	const int noSE = IntegerVector(noSE_R)[0];
	/**** pass arguments from R to cpp *************************************************************************************************************/
	/*#############################################################################################################################################*/
	
	
	
	/*#############################################################################################################################################*/
	/**** some useful constants ********************************************************************************************************************/
	const int n = Y.size();  // number of subjects in the first phase
	const int n2 = X.rows(); // number of subjects in the second phase
	const int n_minus_n2 = n-n2; // number of subjects not selected in the second phase
	const int X_nc = X.cols(); // number of expensive covariates X
	const int ZW_nc = ZW.cols(); // number of inexpensive covariates
	const int ncov = X_nc+ZW_nc; // number of all covariates
	/**** some useful constants ********************************************************************************************************************/
	/*#############################################################################################################################################*/	
	

	
	/*#############################################################################################################################################*/	
	/**** summarize observed distinct rows of X ****************************************************************************************************/
	// m: number of distinct rows of X
	// X_uni_ind(n2): row index of rows of X in X_uni
	// X_uni(m, X_nc): distinct rows of X
	// X_uni_n(m): count of appearances of each distinct row of X	
	int m;
	VectorXi X_index(n2);
	VectorXi X_uni_ind(n2);	
	indexx_Matrix_Row(X, X_index);
	Num_Uni_Matrix_Row(X, X_index, m);	
	MatrixXd X_uni(m, X_nc); 
	VectorXi X_uni_n(m);	
	Create_Uni_Matrix_Row(X, X_index, X_uni, X_uni_ind, X_uni_n);
	/**** summarize observed distinct rows of X ****************************************************************************************************/
	/*#############################################################################################################################################*/
	

		
	/*#############################################################################################################################################*/
	/**** some fixed quantities in the EM algorithm ************************************************************************************************/	
	// t(Y)*Y
	const double LS_YtY_static = Y.squaredNorm(); 
	
	// t(X,ZW)*Y
	VectorXd LS_XtY_static(ncov); 
	LS_XtY_static.head(X_nc) = X.transpose()*Y.head(n2);
	LS_XtY_static.tail(ZW_nc) = ZW.transpose()*Y;
	
	// t(X,ZW)*(X,ZW)
	MatrixXd LS_XtX_static(ncov, ncov); 
	LS_XtX_static.topLeftCorner(X_nc,X_nc) = X.transpose()*X;
	LS_XtX_static.topRightCorner(X_nc,ZW_nc) = X.transpose()*ZW.topRows(n2);
	LS_XtX_static.bottomRightCorner(ZW_nc,ZW_nc) = ZW.transpose()*ZW;	
	/**** some fixed quantities in the EM algorithm ************************************************************************************************/
	/*#############################################################################################################################################*/
	

	
	/*#############################################################################################################################################*/
	/**** output ***********************************************************************************************************************************/
	VectorXd theta(ncov); // regression coefficients
	MatrixXd cov_theta(ncov, ncov); // covariance matrix
	bool flag_nonconvergence; // flag of none convergence in the estimation of regression coefficients
	bool flag_nonconvergence_cov; // flag of none convergence in the estimation of covariance matrix
	double sigma_sq; // residual variance
	/**** output ***********************************************************************************************************************************/
	/*#############################################################################################################################################*/
	

		
	/*#############################################################################################################################################*/
	/**** temporary variables **********************************************************************************************************************/
	VectorXd LS_XtY(ncov);
	MatrixXd LS_XtX(ncov, ncov);
	VectorXd theta0(ncov);
	double sigma_sq0;
	VectorXd p(m); 
	VectorXd p0(m);	
	MatrixXd q(n_minus_n2, m);
	VectorXd q_row_sum(n_minus_n2);
	RowVectorXd q_col_sum(m);
	MatrixXd P_theta(n_minus_n2, m);
	double tol;
	int iter, idx;
	// /* RT's test code */
	// time_t t1, t2;
	// /* RT's test code end */
	/**** temporary variables **********************************************************************************************************************/
	/*#############################################################################################################################################*/
	

	
	/*#############################################################################################################################################*/
	/**** EM algorithm *****************************************************************************************************************************/
	
	/**** parameter initialization *****************************************************************************************************************/
	theta.setZero();
	theta0.setZero();
	sigma_sq = sigma_sq0 = Var(Y);
	for (int k=0; k<m; k++)
	{
		p(k) = (X_uni_n(k)+0.)/(n2+0.);
	}
	p0 = p;
	flag_nonconvergence = false;
	flag_nonconvergence_cov = false;
	/**** parameter initialization *****************************************************************************************************************/
	
	for (iter=0; iter<MAX_ITER; iter++) 
	{
		// /* RT's test code */
		// time(&t1);
		// /* RT's test code end */
		
		/**** E-step *******************************************************************************************************************************/
		
		/**** update P_theta ***********************************************************************************************************************/
		P_theta.col(0) = Y.tail(n_minus_n2);
		P_theta.col(0).noalias() -= ZW.bottomRows(n_minus_n2)*theta.tail(ZW_nc);		
		for (int k=1; k<m; k++) 
		{
			P_theta.col(k) = P_theta.col(0);
		}		
		for (int k=0; k<m; k++) 
		{
			P_theta.col(k).noalias() -= VectorXd::Constant(n_minus_n2, (X_uni.row(k)*theta.head(X_nc))(0,0));	
		}		
		P_theta = P_theta.array().square();
		P_theta /= -2.*sigma_sq;
		P_theta = P_theta.array().exp();
		/**** update P_theta ***********************************************************************************************************************/
		
		/**** update q, q_row_sum ******************************************************************************************************************/
		for (int i=0; i<n_minus_n2; i++) 
		{
			for (int k=0; k<m; k++) 
			{
				q(i,k) = P_theta(i,k)*p(k);
			}
		}
		q_row_sum = q.rowwise().sum();		
		for (int i=0; i<n_minus_n2; i++) 
		{
			q.row(i) /= q_row_sum(i);
		}
		q_col_sum = q.colwise().sum();
		/**** update q, q_row_sum ******************************************************************************************************************/
		
		/**** E-step *******************************************************************************************************************************/
		
		
		/**** M-step *******************************************************************************************************************************/
		
		/**** update theta and sigma_sq ************************************************************************************************************/		
		LS_XtX = LS_XtX_static;
		LS_XtY = LS_XtY_static;
		
		for (int k=0; k<m; k++) 
		{
			LS_XtX.topLeftCorner(X_nc,X_nc).noalias() += q_col_sum(k)*X_uni.row(k).transpose()*X_uni.row(k);
		}
		
		for (int i=0; i<n_minus_n2; i++) 
		{
			idx = i+n2;
			for (int k=0; k<m; k++) 
			{
				LS_XtX.topRightCorner(X_nc,ZW_nc).noalias() += q(i,k)*X_uni.row(k).transpose()*ZW.row(idx);
				LS_XtY.head(X_nc).noalias() += q(i,k)*X_uni.row(k).transpose()*Y(idx);				
			}
		}
		
		theta = LS_XtX.selfadjointView<Eigen::Upper>().ldlt().solve(LS_XtY);		
		sigma_sq = LS_XtY.transpose()*theta;
		sigma_sq = (LS_YtY_static-sigma_sq)/n;
		/**** update theta and sigma_sq ************************************************************************************************************/
		
		/**** update p *****************************************************************************************************************************/
		for (int k=0; k<m; k++) 
		{
			p(k) = X_uni_n(k)+0.;
		}
		p += q_col_sum.transpose();
		p /= n+0.;
		/**** update p *****************************************************************************************************************************/
		
		/**** M-step *******************************************************************************************************************************/
		
		
		/**** calculate the sum of absolute differences between estimates in the current and previous iterations ***********************************/
		tol = (theta-theta0).array().abs().sum();
		tol += fabs(sigma_sq-sigma_sq0);
		tol += (p-p0).array().abs().sum();
		/**** calculate the sum of absolute differences between estimates in the current and previous iterations ***********************************/		
				
		/**** update parameters ********************************************************************************************************************/
		theta0 = theta;
		sigma_sq0 = sigma_sq;
		p0 = p;
		/**** update parameters ********************************************************************************************************************/
		
		/**** check convergence ********************************************************************************************************************/
		if (tol < TOL) 
		{
			break;
		}
		/**** check convergence ********************************************************************************************************************/
		
		// /* RT's test code */
		// time(&t2);
		// Rcout << iter << '\t' << difftime(t2, t1) << '\t' << tol << endl;
		// /* RT's test code end */
	}
	/**** EM algorithm *****************************************************************************************************************************/
	/*#############################################################################################################################################*/
	
	
	
	/*#############################################################################################################################################*/
	/**** variance estimation **********************************************************************************************************************/
	if (iter == MAX_ITER) 
	{
		flag_nonconvergence = true;
		flag_nonconvergence_cov = true;
		theta.setConstant(-999.);
		sigma_sq = -999.;
		cov_theta.setConstant(-999.);
	} 
	else if (noSE) 
	{
		flag_nonconvergence_cov = true;
		cov_theta.setConstant(-999.);
	} 
	else 
	{
		WaldLinearVarianceMLE0(cov_theta, LS_XtX, LS_XtY, p, Y, ZW, theta, X_uni, q, n_minus_n2, X_nc, ZW_nc, m, ncov, n, n2, sigma_sq); 
	}
	/**** variance estimation **********************************************************************************************************************/
	/*#############################################################################################################################################*/
	
	
	
	/*#############################################################################################################################################*/
	/**** return output to R ***********************************************************************************************************************/
	return List::create(Named("theta") = theta,
						Named("sigma_sq") = sigma_sq,
						Named("cov_theta") = cov_theta,
						Named("flag_nonconvergence") = flag_nonconvergence,
						Named("flag_nonconvergence_cov") = flag_nonconvergence_cov);
	/**** return output to R ***********************************************************************************************************************/
	/*#############################################################################################################################################*/
} // TwoPhase_MLE0
