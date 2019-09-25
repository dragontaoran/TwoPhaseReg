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

double WaldLinearMEXYGeneralSplineProfile (Ref<MatrixXd> pB, Ref<RowVectorXd> p_col_sum,
	Ref<VectorXd> q_row_sum, Ref<MatrixXd> p, Ref<MatrixXd> p0, Ref<MatrixXd> P_theta, Ref<MatrixXd> q, Ref<VectorXd> resi_n, Ref<MatrixXd> logp,
	const Ref<const VectorXd>& theta, const Ref<const VectorXd>& Y_tilde, const Ref<const VectorXd>& Y, const Ref<const MatrixXd>& X_tilde, 
	const Ref<const MatrixXd>& X, const Ref<const MatrixXd>& Bspline_uni, const Ref<const MatrixXd>& Z, const Ref<const MatrixXd>& WU_uni, 
	const Ref<const VectorXi>& WU_uni_ind, const Ref<const VectorXi>& Bspline_uni_ind, const Ref<const MatrixXd>& p_static, const double sigma_sq, 
	const int n, const int n2, const int m, const int s, const int n_minus_n2, const int X_nc, const int Z_nc, const int MAX_ITER, const double TOL) 
{
	/**** temporary variables **********************************************************************************************************************/	
	double tol;
	int iter;
	time_t t1, t2;
	/**** temporary variables **********************************************************************************************************************/
	
	/**** update P_theta ***************************************************************************************************************************/	
	P_theta.col(0) = Y_tilde.tail(n_minus_n2);
	P_theta.col(0).noalias() -= X_tilde.bottomRows(n_minus_n2)*theta.head(X_nc);
	P_theta.col(0).noalias() -= Z.bottomRows(n_minus_n2)*theta.tail(Z_nc);
	for (int k=1; k<m; k++) 
	{
		P_theta.col(k) = P_theta.col(0);
	}		
	for (int k=0; k<m; k++) 
	{
		P_theta.col(k).noalias() += VectorXd::Constant(n_minus_n2, -WU_uni(k,0)+(WU_uni.block(k,1,1,X_nc)*theta.head(X_nc))(0,0));
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
		time(&t1);
		
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
		
		time(&t2);
		/* test code */
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
			loglik += pB(Bspline_uni_ind(i),WU_uni_ind(i));
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
		
		resi_n = Y;
		resi_n.noalias() -= X*theta.head(X_nc);
		resi_n.noalias() -= Z.topRows(n2)*theta.tail(Z_nc);
		
		tmp = resi_n.squaredNorm();
		tmp /= 2.*sigma_sq;
		loglik -= tmp;
		/**** calculate the likelihood *************************************************************************************************************/
		
		return loglik;	
	}
} // WaldLinearMEXYGeneralSplineProfile

RcppExport SEXP TwoPhase_MLE0_MEXY (SEXP Y_tilde_R, SEXP X_tilde_R, SEXP Y_R, SEXP X_R, SEXP Z_R, SEXP Bspline_R, SEXP hn_R, SEXP MAX_ITER_R, SEXP TOL_R, SEXP noSE_R) 
{
	/*#############################################################################################################################################*/
	/**** pass arguments from R to cpp *************************************************************************************************************/
	const MapVecd Y_tilde(as<MapVecd>(Y_tilde_R));
	const MapMatd X_tilde(as<MapMatd>(X_tilde_R));
	const MapVecd Y(as<MapVecd>(Y_R));
	const MapMatd X(as<MapMatd>(X_R));
	const MapMatd Z(as<MapMatd>(Z_R));
	const MapMatd Bspline(as<MapMatd>(Bspline_R));
	const double hn = NumericVector(hn_R)[0];
	const int MAX_ITER = IntegerVector(MAX_ITER_R)[0];
	const double TOL = NumericVector(TOL_R)[0];
	const int noSE = IntegerVector(noSE_R)[0];
	/**** pass arguments from R to cpp *************************************************************************************************************/
	/*#############################################################################################################################################*/
	
	
	
	/*#############################################################################################################################################*/
	/**** some useful constants ********************************************************************************************************************/
	const int n = Y_tilde.size();  // number of subjects in the first phase
	const int n2 = Y.size(); // number of subjects in the second phase
	const int n_minus_n2 = n-n2; // number of subjects not selected in the second phase
	const int Z_nc = Z.cols(); // number of inexpensive covariates
	const int X_nc = X.cols(); // number of expensive covariates X
	const int ncov = X_nc+Z_nc; // number of all covariates
	const int s = Bspline.cols(); // number of B-spline functions
	/**** some useful constants ********************************************************************************************************************/
	/*#############################################################################################################################################*/



	/*#############################################################################################################################################*/
	/**** error terms W+U **************************************************************************************************************************/
	const int WU_nc = 1+X_nc;
	MatrixXd WU(n2, WU_nc);
	WU.col(0) = Y_tilde.head(n2)-Y;
	WU.rightCols(X_nc) = X_tilde.topRows(n2)-X;
	/**** error terms W+U **************************************************************************************************************************/
	/*#############################################################################################################################################*/	
	

	
	/*#############################################################################################################################################*/	
	/**** summarize observed distinct rows of WU ***************************************************************************************************/
	// m: number of distinct rows of WU
	// WU_uni_ind(n2): row index of rows of WU in WU_uni
	// WU_uni(m, WU_nc): distinct rows of WU
	// WU_uni_n(m): count of appearances of each distinct row of WU	
	int m;
	VectorXi WU_index(n2);
	VectorXi WU_uni_ind(n2);	
	indexx_Matrix_Row(WU, WU_index);
	Num_Uni_Matrix_Row(WU, WU_index, m);	
	MatrixXd WU_uni(m, WU_nc); 
	VectorXi WU_uni_n(m);	
	Create_Uni_Matrix_Row(WU, WU_index, WU_uni, WU_uni_ind, WU_uni_n);
	/**** summarize observed distinct rows of WU ***************************************************************************************************/
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
	double LS_YtY_static = Y.squaredNorm();
	LS_YtY_static += Y_tilde.tail(n_minus_n2).squaredNorm(); 
	
	// t(X,Z)*Y
	VectorXd LS_XtY_static(ncov); 
	LS_XtY_static.head(X_nc) = X.transpose()*Y;
	LS_XtY_static.head(X_nc).noalias() += X_tilde.bottomRows(n_minus_n2).transpose()*Y_tilde.tail(n_minus_n2);
	LS_XtY_static.tail(Z_nc) = Z.topRows(n2).transpose()*Y;
	LS_XtY_static.tail(Z_nc).noalias() += Z.bottomRows(n_minus_n2).transpose()*Y_tilde.tail(n_minus_n2);
	
	// t(X,Z)*(X,Z)
	MatrixXd LS_XtX_static(ncov, ncov); 
	LS_XtX_static.topLeftCorner(X_nc,X_nc) = X.transpose()*X;
	LS_XtX_static.topLeftCorner(X_nc,X_nc).noalias() += X_tilde.bottomRows(n_minus_n2).transpose()*X_tilde.bottomRows(n_minus_n2);	
	LS_XtX_static.topRightCorner(X_nc,Z_nc) = X.transpose()*Z.topRows(n2);
	LS_XtX_static.topRightCorner(X_nc,Z_nc).noalias() += X_tilde.bottomRows(n_minus_n2).transpose()*Z.bottomRows(n_minus_n2);
	LS_XtX_static.bottomRightCorner(Z_nc,Z_nc) = Z.transpose()*Z;	
	
	// p
	MatrixXd p_static(m, s); 
	p_static.setZero();
	for (int i=0; i<n2; i++) 
	{
		p_static.row(WU_uni_ind(i)) += Bspline.row(i);
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
	double LS_YtY, sigma_sq0;
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
	time_t t1, t2;
	/**** temporary variables **********************************************************************************************************************/
	/*#############################################################################################################################################*/
	

	
	/*#############################################################################################################################################*/
	/**** EM algorithm *****************************************************************************************************************************/
	
	/**** parameter initialization *****************************************************************************************************************/
	theta.setZero();
	theta0.setZero();
	sigma_sq = sigma_sq0 = Var(Y_tilde);

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
		time(&t1);
		
		/**** E-step *******************************************************************************************************************************/
		
		/**** update pB ****************************************************************************************************************************/
		pB = Bspline_uni*p.transpose();
		/**** update pB ****************************************************************************************************************************/
						
		/**** update P_theta ***********************************************************************************************************************/
		P_theta.col(0) = Y_tilde.tail(n_minus_n2);
		P_theta.col(0).noalias() -= X_tilde.bottomRows(n_minus_n2)*theta.head(X_nc);
		P_theta.col(0).noalias() -= Z.bottomRows(n_minus_n2)*theta.tail(Z_nc);
		for (int k=1; k<m; k++) 
		{
			P_theta.col(k) = P_theta.col(0);
		}		
		for (int k=0; k<m; k++) 
		{
			P_theta.col(k).noalias() += VectorXd::Constant(n_minus_n2, -WU_uni(k,0)+(WU_uni.block(k,1,1,X_nc)*theta.head(X_nc))(0,0));
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
		LS_YtY = LS_YtY_static;		
		LS_XtX = LS_XtX_static;
		LS_XtY = LS_XtY_static;
	
		for (int k=0; k<m; k++) 
		{
			LS_YtY += q_col_sum(k)*WU_uni(k,0)*WU_uni(k,0);
			LS_XtX.topLeftCorner(X_nc,X_nc).noalias() += q_col_sum(k)*WU_uni.block(k,1,1,X_nc).transpose()*WU_uni.block(k,1,1,X_nc);
			LS_XtY.head(X_nc).noalias() += q_col_sum(k)*WU_uni.block(k,1,1,X_nc).transpose()*WU_uni(k,0);		
		}
		
		for (int i=0; i<n_minus_n2; i++) 
		{
			idx = i+n2;
			for (int k=0; k<m; k++) 
			{
				LS_YtY -= 2.*q(i,k)*Y_tilde(idx)*WU_uni(k,0);
				LS_XtX.topLeftCorner(X_nc,X_nc).noalias() -= q(i,k)*(WU_uni.block(k,1,1,X_nc).transpose()*X_tilde.row(idx)+X_tilde.row(idx).transpose()*WU_uni.block(k,1,1,X_nc));
				LS_XtX.topRightCorner(X_nc,Z_nc).noalias() -= q(i,k)*WU_uni.block(k,1,1,X_nc).transpose()*Z.row(idx);
				LS_XtY.head(X_nc).noalias() -= q(i,k)*(WU_uni.block(k,1,1,X_nc).transpose()*Y_tilde(idx)+X_tilde.row(idx).transpose()*WU_uni(k,0));
				LS_XtY.tail(Z_nc).noalias() -= q(i,k)*Z.row(idx).transpose()*WU_uni(k,0);
			}
		}
		
		theta = LS_XtX.selfadjointView<Eigen::Upper>().ldlt().solve(LS_XtY);		
		sigma_sq = LS_XtY.transpose()*theta;
		sigma_sq = (LS_YtY-sigma_sq)/n;
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
		
		time(&t2);
		// /* RT's test code */
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
		
		loglik = WaldLinearMEXYGeneralSplineProfile (pB, p_col_sum, q_row_sum, p, p0, P_theta, q, resi_n, logp, theta, Y_tilde, Y, X_tilde, X, Bspline_uni,
			Z, WU_uni, WU_uni_ind, Bspline_uni_ind, p_static, sigma_sq, n, n2, m, s, n_minus_n2, X_nc, Z_nc, MAX_ITER, TOL);
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
			profile_vec(i) = WaldLinearMEXYGeneralSplineProfile (pB, p_col_sum, q_row_sum, p, p0, P_theta, q, resi_n, logp, theta0, Y_tilde, Y, X_tilde, X, Bspline_uni,
				Z, WU_uni, WU_uni_ind, Bspline_uni_ind, p_static, sigma_sq0, n, n2, m, s, n_minus_n2, X_nc, Z_nc, MAX_ITER, TOL);
		}
		theta0 = theta;
		sigma_sq0 = sigma_sq+hn;	
		profile_vec(ncov) = WaldLinearMEXYGeneralSplineProfile (pB, p_col_sum, q_row_sum, p, p0, P_theta, q, resi_n, logp, theta0, Y_tilde, Y, X_tilde, X, Bspline_uni,
			Z, WU_uni, WU_uni_ind, Bspline_uni_ind, p_static, sigma_sq0, n, n2, m, s, n_minus_n2, X_nc, Z_nc, MAX_ITER, TOL);
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
			loglik = WaldLinearMEXYGeneralSplineProfile (pB, p_col_sum, q_row_sum, p, p0, P_theta, q, resi_n, logp, theta0, Y_tilde, Y, X_tilde, X, Bspline_uni,
				Z, WU_uni, WU_uni_ind, Bspline_uni_ind, p_static, sigma_sq0, n, n2, m, s, n_minus_n2, X_nc, Z_nc, MAX_ITER, TOL);
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
				loglik = WaldLinearMEXYGeneralSplineProfile (pB, p_col_sum, q_row_sum, p, p0, P_theta, q, resi_n, logp, theta0, Y_tilde, Y, X_tilde, X, Bspline_uni,
					Z, WU_uni, WU_uni_ind, Bspline_uni_ind, p_static, sigma_sq0, n, n2, m, s, n_minus_n2, X_nc, Z_nc, MAX_ITER, TOL);
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
		loglik = WaldLinearMEXYGeneralSplineProfile (pB, p_col_sum, q_row_sum, p, p0, P_theta, q, resi_n, logp, theta0, Y_tilde, Y, X_tilde, X, Bspline_uni,
			Z, WU_uni, WU_uni_ind, Bspline_uni_ind, p_static, sigma_sq0, n, n2, m, s, n_minus_n2, X_nc, Z_nc, MAX_ITER, TOL);
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
} // TwoPhase_MLE0_MEXY
