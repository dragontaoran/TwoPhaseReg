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

double WaldCoxphGeneralSplineProfile (Ref<MatrixXd> pB, Ref<RowVectorXd> p_col_sum, Ref<VectorXd> ZW_theta, Ref<VectorXd> X_uni_theta, Ref<MatrixXd> e_X_uni_theta,
	Ref<VectorXd> e_X_theta, Ref<VectorXd> lambda, Ref<VectorXd> lambda0, Ref<VectorXd> Lambda,
	Ref<VectorXd> q_row_sum, Ref<MatrixXd> p, Ref<MatrixXd> p0, Ref<MatrixXd> P_theta, Ref<MatrixXd> q, Ref<MatrixXd> logp,
	const Ref<const VectorXd>& theta, const Ref<const VectorXd>& Y, const Ref<const VectorXi>& Delta, const Ref<const MatrixXd>& X, 
	const Ref<const MatrixXd>& Bspline_uni, const Ref<const MatrixXd>& ZW, const Ref<const MatrixXd>& X_uni, const Ref<const VectorXi>& X_uni_ind, 
	const Ref<const VectorXi>& Bspline_uni_ind, const Ref<const VectorXd>& Y_uni_event, const Ref<const VectorXi>& Y_uni_event_n, 
	const Ref<const VectorXi>& Y_risk_ind, const Ref<const MatrixXd>& p_static, const int n, const int n2, const int m, const int n_event_uni, 
	const int s, const int n_minus_n2, const int X_nc, const int ZW_nc, const int MAX_ITER, const double TOL) 
{
	/*#############################################################################################################################################*/
	/**** temporary variables **********************************************************************************************************************/
	double tol;
	int iter, idx;
	// /* RT test block */
	// time_t t1, t2;
	// /* RT test block */
	/**** temporary variables **********************************************************************************************************************/
	/*#############################################################################################################################################*/
		
	/*#############################################################################################################################################*/
	/**** EM algorithm *****************************************************************************************************************************/
	
	/**** fixed quantities *************************************************************************************************************************/
	ZW_theta = ZW*theta.tail(ZW_nc);
	X_uni_theta = X_uni*theta.head(X_nc);
	for (int i=0; i<n_minus_n2; i++)
	{
		for (int k=0; k<m; k++)
		{
			e_X_uni_theta(i,k) = ZW_theta(i+n2)+X_uni_theta(k);
		}
	}
	e_X_uni_theta = e_X_uni_theta.array().exp();
	
	for (int i=0; i<n2; i++)
	{
		e_X_theta(i) = ZW_theta(i)+X_uni_theta(X_uni_ind(i));
	}
	e_X_theta = e_X_theta.array().exp();
	/**** fixed quantities *************************************************************************************************************************/
	
	/**** parameter initialization *****************************************************************************************************************/
	p_col_sum = p_static.colwise().sum();
	for (int j=0; j<s; j++) 
	{
		p.col(j) = p_static.col(j)/p_col_sum(j);
	}
	p0 = p;
		
	lambda.setZero();
	for (int i=0; i<n_event_uni; i++)
	{
		for (int i1=0; i1<n; i1++)
		{
			if (Y(i1) >= Y_uni_event(i))
			{
				lambda(i) ++;
			}
		}
		lambda(i) = (Y_uni_event_n(i)+0.)/lambda(i);
	}
	lambda0 = lambda;
	Lambda(0) = lambda(0);
	for (int i=1; i<n_event_uni; i++)
	{
		Lambda(i) = Lambda(i-1)+lambda(i);
	}
	/**** parameter initialization *****************************************************************************************************************/
	
	for (iter=0; iter<MAX_ITER; iter++) 
	{
		// /* RT test block */
		// time(&t1);
		// /* RT test block */
		
		/**** E-step *******************************************************************************************************************************/
		
		/**** update pB ****************************************************************************************************************************/
		pB = Bspline_uni*p.transpose();
		/**** update pB ****************************************************************************************************************************/
				
		/**** update P_theta ***********************************************************************************************************************/		
		for (int i=0; i<n_minus_n2; i++)
		{
			idx = i+n2;
			if (Y_risk_ind(idx) > -1)
			{
				P_theta.row(i) = -Lambda(Y_risk_ind(idx))*e_X_uni_theta.row(i);
				P_theta.row(i) = P_theta.row(i).array().exp();
				if (Delta(idx) == 1)
				{
					P_theta.row(i) *= lambda(Y_risk_ind(idx));
					P_theta.row(i) = P_theta.row(i).array()*e_X_uni_theta.row(i).array();
				}
			}
			else
			{
				if (Delta(idx) == 1)
				{
					stdError("Error: In WaldCoxphGeneralSplineProfile, the calculation of unique event times is wrong!");
				}
				
				P_theta.row(i).setOnes();			
			}
		}
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
		/**** update q, q_row_sum ******************************************************************************************************************/
		
		/**** E-step *******************************************************************************************************************************/
		
		
		/**** M-step *******************************************************************************************************************************/
		
		/**** update lambda ************************************************************************************************************************/
		lambda.setZero();
		
		for (int i=0; i<n_event_uni; i++)
		{
			for (int i1=0; i1<n2; i1++)
			{
				if (Y(i1) >= Y_uni_event(i))
				{
					lambda(i) += e_X_theta(i1);
				}
			}
			for (int i1=0; i1<n_minus_n2; i1++)
			{
				if (Y(i1+n2) >= Y_uni_event(i))
				{
					for (int k=0; k<m; k++)
					{
						lambda(i) += q(i1,k)*e_X_uni_theta(i1,k);
					}						
				}
			}
			lambda(i) = (Y_uni_event_n(i)+0.)/lambda(i);
		}
		Lambda(0) = lambda(0);
		for (int i=1; i<n_event_uni; i++)
		{
			Lambda(i) = Lambda(i-1)+lambda(i);
		}		
		/**** update lambda ************************************************************************************************************************/
		
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
		tol = (lambda-lambda0).array().abs().sum();
		tol += (p-p0).array().abs().sum();
		/**** calculate the sum of absolute differences between estimates in the current and previous iterations ***********************************/		
				
		/**** update parameters ********************************************************************************************************************/
		lambda0 = lambda;
		p0 = p;
		/**** update parameters ********************************************************************************************************************/
		
		/**** check convergence ********************************************************************************************************************/
		if (tol < TOL) 
		{
			break;
		}
		/**** check convergence ********************************************************************************************************************/

		// /* RT test block */
		// time(&t2);
		// Rcout << iter << '\t' << difftime(t2, t1) << '\t' << tol << endl;
		// /* RT test block */
	}
	/**** EM algorithm *****************************************************************************************************************************/
	/*#############################################################################################################################################*/

	if (iter == MAX_ITER) 
	{
		return -999.;
	} 
	else 
	{
		/**** calculate the likelihood *************************************************************************************************************/
		double loglik;
		
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
			loglik -= Lambda(Y_risk_ind(i))*e_X_theta(i);
			if (Delta(i) == 1)
			{
				loglik += log(lambda(Y_risk_ind(i)))+ZW_theta(i)+X_uni_theta(X_uni_ind(i));
			}
		}
		
		pB = Bspline_uni*p.transpose();;
		
		for (int i=0; i<n_minus_n2; i++) 
		{
			for (int k=0; k<m; k++) 
			{
				q(i,k) = P_theta(i,k)*pB(Bspline_uni_ind(i+n2),k);
			}
		}
		q_row_sum = q.rowwise().sum();		
		
		loglik += q_row_sum.array().log().sum();	
		/**** calculate the likelihood *************************************************************************************************************/
		
		return loglik;	
	}
} // WaldCoxphGeneralSplineProfile

RcppExport SEXP TwoPhase_GeneralSpline_coxph (SEXP Y_R, SEXP Delta_R, SEXP X_R, SEXP ZW_R, SEXP Bspline_R, SEXP hn_R, SEXP MAX_ITER_R, SEXP TOL_R, SEXP noSE_R) 
{
	/*#############################################################################################################################################*/
	/**** pass arguments from R to cpp *************************************************************************************************************/
	const MapVecd Y(as<MapVecd>(Y_R));
	const MapVeci Delta(as<MapVeci>(Delta_R));
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
	/**** summarize observed distinct values of Y **************************************************************************************************/
	// n_event_uni: number of distinct values of Y where Delta=1
	// Y_uni_event: vector of unique events
	// Y_risk_ind: vector of indexes of which one of the risk sets each element in Y corresponds to.
	// Y_uni_event_n: count of appearances of each distinct event time	
	int n_event_uni;
	VectorXi Y_index(n);	
	indexx_Vector(Y, Y_index);
	Num_Distinct_Events(Y, Y_index, Delta, n_event_uni);
	VectorXd Y_uni_event(n_event_uni); 
	VectorXi Y_risk_ind(n);
	VectorXi Y_uni_event_n(n_event_uni);
	Create_Uni_Events(Y, Y_index, Delta, Y_uni_event, Y_risk_ind, Y_uni_event_n);
	/**** summarize observed distinct values of Y **************************************************************************************************/
	/*#############################################################################################################################################*/
	
	
	
	/*#############################################################################################################################################*/
	/**** some fixed quantities in the EM algorithm ************************************************************************************************/		
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
	/**** output ***********************************************************************************************************************************/
	/*#############################################################################################################################################*/
	

		
	/*#############################################################################################################################################*/
	/**** temporary variables **********************************************************************************************************************/
	VectorXd LS_XtY(ncov);
	MatrixXd LS_XtX(ncov, ncov);
	VectorXd theta0(ncov);
	MatrixXd p(m, s); 
	MatrixXd p0(m, s);
	RowVectorXd p_col_sum(s);	
	MatrixXd q(n_minus_n2, m);
	VectorXd q_row_sum(n_minus_n2);
	MatrixXd pB(m_B, m);
	MatrixXd P_theta(n_minus_n2, m);
	double tol;
	int iter, idx, idx1;
	VectorXd ZW_theta(n);
	VectorXd X_uni_theta(m);
	MatrixXd e_X_uni_theta(n_minus_n2, m);
	VectorXd e_X_theta(n2);
	string error_message;
	double s0;
	VectorXd s1(ncov);
	MatrixXd s2(ncov, ncov);
	VectorXd lambda(n_event_uni);
	VectorXd lambda0(n_event_uni);
	VectorXd Lambda(n_event_uni);
	// /* RT test block */
	// time_t t1, t2;
	// /* RT test block */
	/**** temporary variables **********************************************************************************************************************/
	/*#############################################################################################################################################*/
	

	
	/*#############################################################################################################################################*/
	/**** EM algorithm *****************************************************************************************************************************/
	
	/**** parameter initialization *****************************************************************************************************************/
	theta.setZero();
	theta0.setZero();
	
	p_col_sum = p_static.colwise().sum();
	for (int j=0; j<s; j++) 
	{
		p.col(j) = p_static.col(j)/p_col_sum(j);
	}
	p0 = p;
	
	flag_nonconvergence = false;
	flag_nonconvergence_cov = false;
	
	lambda.setZero();
	for (int i=0; i<n_event_uni; i++)
	{
		for (int i1=0; i1<n; i1++)
		{
			if (Y(i1) >= Y_uni_event(i))
			{
				lambda(i) ++;
			}
		}
		lambda(i) = (Y_uni_event_n(i)+0.)/lambda(i);
	}
	lambda0 = lambda;
	Lambda(0) = lambda(0);
	for (int i=1; i<n_event_uni; i++)
	{
		Lambda(i) = Lambda(i-1)+lambda(i);
	}
	/**** parameter initialization *****************************************************************************************************************/
	
	for (iter=0; iter<MAX_ITER; iter++) 
	{
		// /* RT test block */
		// time(&t1);
		// /* RT test block */
		
		/**** E-step *******************************************************************************************************************************/
		
		/**** update pB ****************************************************************************************************************************/
		pB = Bspline_uni*p.transpose();
		/**** update pB ****************************************************************************************************************************/
				
		/**** update P_theta ***********************************************************************************************************************/
		ZW_theta = ZW*theta.tail(ZW_nc);
		X_uni_theta = X_uni*theta.head(X_nc);
		for (int i=0; i<n_minus_n2; i++)
		{
			for (int k=0; k<m; k++)
			{
				e_X_uni_theta(i,k) = ZW_theta(i+n2)+X_uni_theta(k);
			}
		}
		e_X_uni_theta = e_X_uni_theta.array().exp();
		
		for (int i=0; i<n_minus_n2; i++)
		{
			idx = i+n2;
			if (Y_risk_ind(idx) > -1)
			{
				P_theta.row(i) = -Lambda(Y_risk_ind(idx))*e_X_uni_theta.row(i);
				P_theta.row(i) = P_theta.row(i).array().exp();
				if (Delta(idx) == 1)
				{
					P_theta.row(i) *= lambda(Y_risk_ind(idx));
					P_theta.row(i) = P_theta.row(i).array()*e_X_uni_theta.row(i).array();
				}
			}
			else
			{
				if (Delta(idx) == 1)
				{
					stdError("Error: In TwoPhase_GeneralSpline_coxph, the calculation of unique event times is wrong!");
				}
				
				P_theta.row(i).setOnes();			
			}
		}
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
		/**** update q, q_row_sum ******************************************************************************************************************/
		
		/**** E-step *******************************************************************************************************************************/
		
		
		/**** M-step *******************************************************************************************************************************/
		
		/**** update theta *************************************************************************************************************************/		
		for (int i=0; i<n2; i++)
		{
			e_X_theta(i) = ZW_theta(i)+X_uni_theta(X_uni_ind(i));
		}
		e_X_theta = e_X_theta.array().exp();
		
		LS_XtX.setZero();
		LS_XtY.setZero();
		for (int i=0; i<n2; i++)
		{
			if (Delta(i) == 1)
			{
				s0 = 0.;
				s1.setZero();
				s2.setZero();
				
				for (int i1=0; i1<n2; i1++)
				{
					if (Y(i1) >= Y(i))
					{
						s0 += e_X_theta(i1);
						s1.head(X_nc).noalias() += X.row(i1).transpose()*e_X_theta(i1);
						s1.tail(ZW_nc).noalias() += ZW.row(i1).transpose()*e_X_theta(i1);						
						s2.topLeftCorner(X_nc,X_nc).noalias() += X.row(i1).transpose()*X.row(i1)*e_X_theta(i1);
						s2.topRightCorner(X_nc,ZW_nc).noalias() += X.row(i1).transpose()*ZW.row(i1)*e_X_theta(i1);
						s2.bottomRightCorner(ZW_nc,ZW_nc).noalias() += ZW.row(i1).transpose()*ZW.row(i1)*e_X_theta(i1);
					}
				}
				for (int i1=0; i1<n_minus_n2; i1++)
				{
					idx1 = i1+n2;
					if (Y(idx1) >= Y(i))
					{
						for (int k=0; k<m; k++)
						{
							s0 += q(i1,k)*e_X_uni_theta(i1,k);
							s1.head(X_nc).noalias() += q(i1,k)*X_uni.row(k).transpose()*e_X_uni_theta(i1,k);
							s1.tail(ZW_nc).noalias() += q(i1,k)*ZW.row(idx1).transpose()*e_X_uni_theta(i1,k);
							s2.topLeftCorner(X_nc,X_nc).noalias() += q(i1,k)*X_uni.row(k).transpose()*X_uni.row(k)*e_X_uni_theta(i1,k);
							s2.topRightCorner(X_nc,ZW_nc).noalias() += q(i1,k)*X_uni.row(k).transpose()*ZW.row(idx1)*e_X_uni_theta(i1,k);
							s2.bottomRightCorner(ZW_nc,ZW_nc).noalias() += q(i1,k)*ZW.row(idx1).transpose()*ZW.row(idx1)*e_X_uni_theta(i1,k);
						}
					}					
				}
				LS_XtY.head(X_nc).noalias() += X.row(i).transpose()-s1.head(X_nc)/s0;
				LS_XtY.tail(ZW_nc).noalias() += ZW.row(i).transpose()-s1.tail(ZW_nc)/s0;
				LS_XtX.noalias() += s2/s0-s1*s1.transpose()/(s0*s0);
			}
		}

		for (int i=0; i<n_minus_n2; i++)
		{
			idx = i+n2;
			if (Delta(idx) == 1)
			{
				s0 = 0.;
				s1.setZero();
				s2.setZero();
				
				for (int i1=0; i1<n2; i1++)
				{
					if (Y(i1) >= Y(idx))
					{
						s0 += e_X_theta(i1);
						s1.head(X_nc).noalias() += X.row(i1).transpose()*e_X_theta(i1);
						s1.tail(ZW_nc).noalias() += ZW.row(i1).transpose()*e_X_theta(i1);						
						s2.topLeftCorner(X_nc,X_nc).noalias() += X.row(i1).transpose()*X.row(i1)*e_X_theta(i1);
						s2.topRightCorner(X_nc,ZW_nc).noalias() += X.row(i1).transpose()*ZW.row(i1)*e_X_theta(i1);
						s2.bottomRightCorner(ZW_nc,ZW_nc).noalias() += ZW.row(i1).transpose()*ZW.row(i1)*e_X_theta(i1);
					}
				}
				
				for (int i1=0; i1<n_minus_n2; i1++)
				{
					idx1 = i1+n2;
					if (Y(idx1) >= Y(idx))
					{
						for (int k=0; k<m; k++)
						{
							s0 += q(i1,k)*e_X_uni_theta(i1,k);							
							s1.head(X_nc).noalias() += q(i1,k)*X_uni.row(k).transpose()*e_X_uni_theta(i1,k);
							s1.tail(ZW_nc).noalias() += q(i1,k)*ZW.row(idx1).transpose()*e_X_uni_theta(i1,k);						
							s2.topLeftCorner(X_nc,X_nc).noalias() += q(i1,k)*X_uni.row(k).transpose()*X_uni.row(k)*e_X_uni_theta(i1,k);
							s2.topRightCorner(X_nc,ZW_nc).noalias() += q(i1,k)*X_uni.row(k).transpose()*ZW.row(idx1)*e_X_uni_theta(i1,k);
							s2.bottomRightCorner(ZW_nc,ZW_nc).noalias() += q(i1,k)*ZW.row(idx1).transpose()*ZW.row(idx1)*e_X_uni_theta(i1,k);							
						}
					}					
				}				
				for (int k=0; k<m; k++)
				{
					LS_XtY.head(X_nc).noalias() += q(i,k)*X_uni.row(k).transpose();				
				}
				LS_XtY.head(X_nc).noalias() -= s1.head(X_nc)/s0;			
				LS_XtY.tail(ZW_nc).noalias() += ZW.row(idx).transpose()-s1.tail(ZW_nc)/s0;
				LS_XtX.noalias() += s2/s0-s1*s1.transpose()/(s0*s0);
			}
		}

		theta = LS_XtX.selfadjointView<Eigen::Upper>().ldlt().solve(LS_XtY);
		theta += theta0;
		/**** update theta *************************************************************************************************************************/
		
		/**** update lambda ************************************************************************************************************************/
		lambda.setZero();
		
		for (int i=0; i<n_event_uni; i++)
		{
			for (int i1=0; i1<n2; i1++)
			{
				if (Y(i1) >= Y_uni_event(i))
				{
					lambda(i) += e_X_theta(i1);
				}
			}
			for (int i1=0; i1<n_minus_n2; i1++)
			{
				if (Y(i1+n2) >= Y_uni_event(i))
				{
					for (int k=0; k<m; k++)
					{
						lambda(i) += q(i1,k)*e_X_uni_theta(i1,k);
					}						
				}
			}
			lambda(i) = (Y_uni_event_n(i)+0.)/lambda(i);
		}
		Lambda(0) = lambda(0);
		for (int i=1; i<n_event_uni; i++)
		{
			Lambda(i) = Lambda(i-1)+lambda(i);
		}		
		/**** update lambda ************************************************************************************************************************/
		
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
		tol += (lambda-lambda0).array().abs().sum();
		tol += (p-p0).array().abs().sum();
		/**** calculate the sum of absolute differences between estimates in the current and previous iterations ***********************************/		
				
		/**** update parameters ********************************************************************************************************************/
		theta0 = theta;
		lambda0 = lambda;
		p0 = p;
		/**** update parameters ********************************************************************************************************************/
		
		/**** check convergence ********************************************************************************************************************/
		if (tol < TOL) 
		{
			break;
		}
		/**** check convergence ********************************************************************************************************************/

		// /* RT test block */
		// time(&t2);
		// Rcout << iter << '\t' << difftime(t2, t1) << '\t' << tol << endl;
		// /* RT test block */
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
		cov_theta.setConstant(-999.);
	} 
	else if (noSE) 
	{
		flag_nonconvergence_cov = true;
		cov_theta.setConstant(-999.);
	} 
	else 
	{
		VectorXd profile_vec(ncov);
		MatrixXd profile_mat(ncov, ncov);
		MatrixXd logp(m, s);
		MatrixXd inv_profile_mat(ncov, ncov);
		double loglik;
				
		profile_mat.setZero();
		profile_vec.setZero();
		
		loglik = WaldCoxphGeneralSplineProfile(pB, p_col_sum, ZW_theta, X_uni_theta, e_X_uni_theta, e_X_theta, lambda, lambda0, Lambda, q_row_sum, p, p0, P_theta, q, logp,
			theta, Y, Delta, X, Bspline_uni, ZW, X_uni, X_uni_ind, Bspline_uni_ind, Y_uni_event, Y_uni_event_n, 
			Y_risk_ind, p_static, n, n2, m, n_event_uni, s, n_minus_n2, X_nc, ZW_nc, MAX_ITER, TOL);

		if (loglik == -999.) 
		{
			flag_nonconvergence_cov = true;
		}
		for (int i=0; i<ncov; i++) 
		{
			for (int j=i; j<ncov; j++) 
			{
				profile_mat(i,j) = loglik;
			}
		}

		for (int i=0; i<ncov; i++) 
		{
			theta0 = theta;
			theta0(i) += hn;
			profile_vec(i) = WaldCoxphGeneralSplineProfile(pB, p_col_sum, ZW_theta, X_uni_theta, e_X_uni_theta, e_X_theta, lambda, lambda0, Lambda, q_row_sum, p, p0, P_theta, q, logp,
				theta0, Y, Delta, X, Bspline_uni, ZW, X_uni, X_uni_ind, Bspline_uni_ind, Y_uni_event, Y_uni_event_n, 
				Y_risk_ind, p_static, n, n2, m, n_event_uni, s, n_minus_n2, X_nc, ZW_nc, MAX_ITER, TOL);
		}
		for (int i=0; i<ncov; i++) 
		{
			if(profile_vec(i) == -999.) 
			{
				flag_nonconvergence_cov = true;
			}
		}
		
		for (int i=0; i<ncov; i++) 
		{
			for (int j=i; j<ncov; j++) 
			{
				theta0 = theta;
				theta0(i) += hn;
				theta0(j) += hn;
				loglik = WaldCoxphGeneralSplineProfile(pB, p_col_sum, ZW_theta, X_uni_theta, e_X_uni_theta, e_X_theta, lambda, lambda0, Lambda, q_row_sum, p, p0, P_theta, q, logp,
					theta0, Y, Delta, X, Bspline_uni, ZW, X_uni, X_uni_ind, Bspline_uni_ind, Y_uni_event, Y_uni_event_n, 
					Y_risk_ind, p_static, n, n2, m, n_event_uni, s, n_minus_n2, X_nc, ZW_nc, MAX_ITER, TOL);
				if (loglik == -999.) 
				{
					flag_nonconvergence_cov = true;
				}				
				profile_mat(i,j) += loglik;
				profile_mat(i,j) -= profile_vec(i)+profile_vec(j);
			}
		}
				
		if (flag_nonconvergence_cov == true) 
		{
			cov_theta.setConstant(-999.);
		} 
		else 
		{		
			for (int i=0; i<ncov; i++) 
			{
				for (int j=i+1; j<ncov; j++) 
				{
					profile_mat(j,i) = profile_mat(i,j);
				}
			}		
			profile_mat /= hn*hn;
			profile_mat = -profile_mat;

			inv_profile_mat = profile_mat.selfadjointView<Eigen::Upper>().ldlt().solve(MatrixXd::Identity(ncov, ncov));
			cov_theta = inv_profile_mat.topLeftCorner(ncov,ncov);
		}
	}
	/**** variance estimation **********************************************************************************************************************/
	/*#############################################################################################################################################*/
	
	
	
	/*#############################################################################################################################################*/
	/**** return output to R ***********************************************************************************************************************/
	return List::create(Named("theta") = theta,
						Named("cov_theta") = cov_theta,
						Named("flag_nonconvergence") = flag_nonconvergence,
						Named("flag_nonconvergence_cov") = flag_nonconvergence_cov);
	/**** return output to R ***********************************************************************************************************************/
	/*#############################################################################################################################################*/
} // TwoPhase_GeneralSpline_coxph

RcppExport SEXP TwoPhase_MLE0_coxph (SEXP Y_R, SEXP Delta_R, SEXP X_R, SEXP ZW_R, SEXP MAX_ITER_R, SEXP TOL_R, SEXP noSE_R) 
{
	/*#############################################################################################################################################*/
	/**** pass arguments from R to cpp *************************************************************************************************************/
	const MapVecd Y(as<MapVecd>(Y_R));
	const MapVeci Delta(as<MapVeci>(Delta_R));
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
	const int ZW_nc = ZW.cols(); // number of inexpensive covariates
	const int X_nc = X.cols(); // number of expensive covariates X
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
	/**** summarize observed distinct values of Y **************************************************************************************************/
	// n_event_uni: number of distinct values of Y where Delta=1
	// Y_uni_event: vector of unique events
	// Y_risk_ind: vector of indexes of which one of the risk sets each element in Y corresponds to.
	// Y_uni_event_n: count of appearances of each distinct event time	
	int n_event_uni;
	VectorXi Y_index(n);	
	indexx_Vector(Y, Y_index);
	Num_Distinct_Events(Y, Y_index, Delta, n_event_uni); 	
	VectorXd Y_uni_event(n_event_uni); 
	VectorXi Y_risk_ind(n);
	VectorXi Y_uni_event_n(n_event_uni);
	Create_Uni_Events(Y, Y_index, Delta, Y_uni_event, Y_risk_ind, Y_uni_event_n);
	/**** summarize observed distinct values of Y **************************************************************************************************/
	/*#############################################################################################################################################*/
	
	
	
	/*#############################################################################################################################################*/
	/**** output ***********************************************************************************************************************************/
	VectorXd theta(ncov); // regression coefficients
	MatrixXd cov_theta(ncov, ncov); // covariance matrix
	bool flag_nonconvergence; // flag of none convergence in the estimation of regression coefficients
	bool flag_nonconvergence_cov; // flag of none convergence in the estimation of covariance matrix
	/**** output ***********************************************************************************************************************************/
	/*#############################################################################################################################################*/
	

		
	/*#############################################################################################################################################*/
	/**** temporary variables **********************************************************************************************************************/
	VectorXd LS_XtY(ncov);
	MatrixXd LS_XtX(ncov, ncov);
	VectorXd theta0(ncov);
	VectorXd p(m); 
	VectorXd p0(m);		
	MatrixXd q(n_minus_n2, m);
	VectorXd q_row_sum(n_minus_n2);
	RowVectorXd q_col_sum(m);
	MatrixXd P_theta(n_minus_n2, m);
	double tol;
	int iter, idx, idx1;
	VectorXd ZW_theta(n);
	VectorXd X_uni_theta(m);
	MatrixXd e_X_uni_theta(n_minus_n2, m);
	VectorXd e_X_theta(n2);
	string error_message;
	double s0;
	VectorXd s1(ncov);
	MatrixXd s2(ncov, ncov);
	VectorXd lambda(n_event_uni);
	VectorXd lambda0(n_event_uni);
	VectorXd Lambda(n_event_uni);
	// /* RT test block */
	// time_t t1, t2;
	// /* RT test block */
	/**** temporary variables **********************************************************************************************************************/
	/*#############################################################################################################################################*/
	

	
	/*#############################################################################################################################################*/
	/**** EM algorithm *****************************************************************************************************************************/
	
	/**** parameter initialization *****************************************************************************************************************/
	theta.setZero();
	theta0.setZero();
		
	for (int k=0; k<m; k++)
	{
		p(k) = (X_uni_n(k)+0.)/(n2+0.);
	}
	p0 = p;
	
	flag_nonconvergence = false;
	flag_nonconvergence_cov = false;
	
	lambda.setZero();
	for (int i=0; i<n_event_uni; i++)
	{
		for (int i1=0; i1<n; i1++)
		{
			if (Y(i1) >= Y_uni_event(i))
			{
				lambda(i) ++;
			}
		}
		lambda(i) = (Y_uni_event_n(i)+0.)/lambda(i);
	}
	lambda0 = lambda;
	Lambda(0) = lambda(0);
	for (int i=1; i<n_event_uni; i++)
	{
		Lambda(i) = Lambda(i-1)+lambda(i);
	}
	/**** parameter initialization *****************************************************************************************************************/
	
	for (iter=0; iter<MAX_ITER; iter++) 
	{
		// /* RT test block */
		//	time(&t1);
		//	/* RT test block */
		
		/**** E-step *******************************************************************************************************************************/
		
		/**** update P_theta ***********************************************************************************************************************/
		ZW_theta = ZW*theta.tail(ZW_nc);
		X_uni_theta = X_uni*theta.head(X_nc);
		for (int i=0; i<n_minus_n2; i++)
		{
			for (int k=0; k<m; k++)
			{
				e_X_uni_theta(i,k) = ZW_theta(i+n2)+X_uni_theta(k);
			}
		}
		e_X_uni_theta = e_X_uni_theta.array().exp();
		
		for (int i=0; i<n_minus_n2; i++)
		{
			idx = i+n2;
			if (Y_risk_ind(idx) > -1)
			{
				P_theta.row(i) = -Lambda(Y_risk_ind(idx))*e_X_uni_theta.row(i);
				P_theta.row(i) = P_theta.row(i).array().exp();
				if (Delta(idx) == 1)
				{
					P_theta.row(i) *= lambda(Y_risk_ind(idx));
					P_theta.row(i) = P_theta.row(i).array()*e_X_uni_theta.row(i).array();
				}				
			}
			else
			{
				if (Delta(idx) == 1)
				{
					stdError("Error: In TwoPhase_MLE0_coxph, the calculation of unique event times is wrong!");
				}
				
				P_theta.row(i).setOnes();			
			}
		}
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
		
		/**** update theta *************************************************************************************************************************/		
		for (int i=0; i<n2; i++)
		{
			e_X_theta(i) = ZW_theta(i)+X_uni_theta(X_uni_ind(i));
		}
		e_X_theta = e_X_theta.array().exp();

		LS_XtX.setZero();
		LS_XtY.setZero();
		
		for (int i=0; i<n2; i++)
		{
			if (Delta(i) == 1)
			{
				s0 = 0.;
				s1.setZero();
				s2.setZero();
				
				for (int i1=0; i1<n2; i1++)
				{
					if (Y(i1) >= Y(i))
					{
						s0 += e_X_theta(i1);						
						s1.head(X_nc).noalias() += X.row(i1).transpose()*e_X_theta(i1);
						s1.tail(ZW_nc).noalias() += ZW.row(i1).transpose()*e_X_theta(i1);				
						s2.topLeftCorner(X_nc,X_nc).noalias() += X.row(i1).transpose()*X.row(i1)*e_X_theta(i1);
						s2.topRightCorner(X_nc,ZW_nc).noalias() += X.row(i1).transpose()*ZW.row(i1)*e_X_theta(i1);
						s2.bottomRightCorner(ZW_nc,ZW_nc).noalias() += ZW.row(i1).transpose()*ZW.row(i1)*e_X_theta(i1);
					}
				}
				
				for (int i1=0; i1<n_minus_n2; i1++)
				{
					idx1 = i1+n2;
					if (Y(idx1) >= Y(i))
					{
						for (int k=0; k<m; k++)
						{
							s0 += q(i1,k)*e_X_uni_theta(i1,k);						
							s1.head(X_nc).noalias() += q(i1,k)*X_uni.row(k).transpose()*e_X_uni_theta(i1,k);
							s1.tail(ZW_nc).noalias() += q(i1,k)*ZW.row(idx1).transpose()*e_X_uni_theta(i1,k);
							s2.topLeftCorner(X_nc,X_nc).noalias() += q(i1,k)*X_uni.row(k).transpose()*X_uni.row(k)*e_X_uni_theta(i1,k);
							s2.topRightCorner(X_nc,ZW_nc).noalias() += q(i1,k)*X_uni.row(k).transpose()*ZW.row(idx1)*e_X_uni_theta(i1,k);
							s2.bottomRightCorner(ZW_nc,ZW_nc).noalias() += q(i1,k)*ZW.row(idx1).transpose()*ZW.row(idx1)*e_X_uni_theta(i1,k);							
						}
					}					
				}				
				LS_XtY.head(X_nc).noalias() += X.row(i).transpose()-s1.head(X_nc)/s0;
				LS_XtY.tail(ZW_nc).noalias() += ZW.row(i).transpose()-s1.tail(ZW_nc)/s0;
				LS_XtX.noalias() += s2/s0-s1*s1.transpose()/(s0*s0);
			}
		}
		
		for (int i=0; i<n_minus_n2; i++)
		{
			idx = i+n2;
			if (Delta(idx) == 1)
			{
				s0 = 0.;
				s1.setZero();
				s2.setZero();
				
				for (int i1=0; i1<n2; i1++)
				{
					if (Y(i1) >= Y(idx))
					{
						s0 += e_X_theta(i1);					
						s1.head(X_nc).noalias() += X.row(i1).transpose()*e_X_theta(i1);
						s1.tail(ZW_nc).noalias() += ZW.row(i1).transpose()*e_X_theta(i1);				
						s2.topLeftCorner(X_nc,X_nc).noalias() += X.row(i1).transpose()*X.row(i1)*e_X_theta(i1);
						s2.topRightCorner(X_nc,ZW_nc).noalias() += X.row(i1).transpose()*ZW.row(i1)*e_X_theta(i1);
						s2.bottomRightCorner(ZW_nc,ZW_nc).noalias() += ZW.row(i1).transpose()*ZW.row(i1)*e_X_theta(i1);
					}
				}
				
				for (int i1=0; i1<n_minus_n2; i1++)
				{
					idx1 = i1+n2;
					if (Y(idx1) >= Y(idx))
					{
						for (int k=0; k<m; k++)
						{
							s0 += q(i1,k)*e_X_uni_theta(i1,k);						
							s1.head(X_nc).noalias() += q(i1,k)*X_uni.row(k).transpose()*e_X_uni_theta(i1,k);
							s1.tail(ZW_nc).noalias() += q(i1,k)*ZW.row(idx1).transpose()*e_X_uni_theta(i1,k);						
							s2.topLeftCorner(X_nc,X_nc).noalias() += q(i1,k)*X_uni.row(k).transpose()*X_uni.row(k)*e_X_uni_theta(i1,k);
							s2.topRightCorner(X_nc,ZW_nc).noalias() += q(i1,k)*X_uni.row(k).transpose()*ZW.row(idx1)*e_X_uni_theta(i1,k);
							s2.bottomRightCorner(ZW_nc,ZW_nc).noalias() += q(i1,k)*ZW.row(idx1).transpose()*ZW.row(idx1)*e_X_uni_theta(i1,k);							
						}
					}					
				}			
				for (int k=0; k<m; k++)
				{
					LS_XtY.head(X_nc).noalias() += q(i,k)*X_uni.row(k).transpose();				
				}
				LS_XtY.head(X_nc).noalias() -= s1.head(X_nc)/s0;			
				LS_XtY.tail(ZW_nc).noalias() += ZW.row(idx).transpose()-s1.tail(ZW_nc)/s0;
				LS_XtX.noalias() += s2/s0-s1*s1.transpose()/(s0*s0);
			}
		}
		
		theta = LS_XtX.selfadjointView<Eigen::Upper>().ldlt().solve(LS_XtY);
		theta += theta0;
		/**** update theta *************************************************************************************************************************/
		
		/**** update lambda ************************************************************************************************************************/
		lambda.setZero();
		
		for (int i=0; i<n_event_uni; i++)
		{
			for (int i1=0; i1<n2; i1++)
			{
				if (Y(i1) >= Y_uni_event(i))
				{
					lambda(i) += e_X_theta(i1);
				}
			}
			for (int i1=0; i1<n_minus_n2; i1++)
			{
				if (Y(i1+n2) >= Y_uni_event(i))
				{
					for (int k=0; k<m; k++)
					{
						lambda(i) += q(i1,k)*e_X_uni_theta(i1,k);
					}						
				}
			}
			lambda(i) = (Y_uni_event_n(i)+0.)/lambda(i);
		}

		Lambda(0) = lambda(0);
		for (int i=1; i<n_event_uni; i++)
		{
			Lambda(i) = Lambda(i-1)+lambda(i);
		}		
		/**** update lambda ************************************************************************************************************************/
		
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
		tol += (lambda-lambda0).array().abs().sum();
		tol += (p-p0).array().abs().sum();
		/**** calculate the sum of absolute differences between estimates in the current and previous iterations ***********************************/		
				
		/**** update parameters ********************************************************************************************************************/
		theta0 = theta;
		lambda0 = lambda;
		p0 = p;
		/**** update parameters ********************************************************************************************************************/
		
		/**** check convergence ********************************************************************************************************************/
		if (tol < TOL) 
		{
			break;
		}
		/**** check convergence ********************************************************************************************************************/
		
		// /* RT test block */
		//	time(&t2);
		//	Rcout << iter << '\t' << difftime(t2, t1) << '\t' << tol << endl;
		// /* RT test block */
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
		cov_theta.setConstant(-999.);
	} 
	else if (noSE) 
	{
		flag_nonconvergence_cov = true;
		cov_theta.setConstant(-999.);
	} 
	else 
	{
		// to be added
	}
	/**** variance estimation **********************************************************************************************************************/
	/*#############################################################################################################################################*/
	
	
	
	/*#############################################################################################################################################*/
	/**** return output to R ***********************************************************************************************************************/
	return List::create(Named("theta") = theta,
						Named("cov_theta") = cov_theta,
						Named("flag_nonconvergence") = flag_nonconvergence,
						Named("flag_nonconvergence_cov") = flag_nonconvergence_cov);
	/**** return output to R ***********************************************************************************************************************/
	/*#############################################################################################################################################*/
} // TwoPhase_MLE0_coxph

RcppExport SEXP TwoPhase_MLE0_noZW_coxph (SEXP Y_R, SEXP Delta_R, SEXP X_R, SEXP MAX_ITER_R, SEXP TOL_R, SEXP noSE_R) 
{
	/*#############################################################################################################################################*/
	/**** pass arguments from R to cpp *************************************************************************************************************/
	const MapVecd Y(as<MapVecd>(Y_R));
	const MapVeci Delta(as<MapVeci>(Delta_R));
	const MapMatd X(as<MapMatd>(X_R));
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
	const int ncov = X_nc;
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
	/**** summarize observed distinct values of Y **************************************************************************************************/
	// n_event_uni: number of distinct values of Y where Delta=1
	// Y_uni_event: vector of unique events
	// Y_risk_ind: vector of indexes of which one of the risk sets each element in Y corresponds to.
	// Y_uni_event_n: count of appearances of each distinct event time	
	int n_event_uni;
	VectorXi Y_index(n);	
	indexx_Vector(Y, Y_index);
	Num_Distinct_Events(Y, Y_index, Delta, n_event_uni); 	
	VectorXd Y_uni_event(n_event_uni); 
	VectorXi Y_risk_ind(n);
	VectorXi Y_uni_event_n(n_event_uni);
	Create_Uni_Events(Y, Y_index, Delta, Y_uni_event, Y_risk_ind, Y_uni_event_n);
	/**** summarize observed distinct values of Y **************************************************************************************************/
	/*#############################################################################################################################################*/
	
	
	
	/*#############################################################################################################################################*/
	/**** output ***********************************************************************************************************************************/
	VectorXd theta(ncov); // regression coefficients
	MatrixXd cov_theta(ncov, ncov); // covariance matrix
	bool flag_nonconvergence; // flag of none convergence in the estimation of regression coefficients
	bool flag_nonconvergence_cov; // flag of none convergence in the estimation of covariance matrix
	/**** output ***********************************************************************************************************************************/
	/*#############################################################################################################################################*/
	

		
	/*#############################################################################################################################################*/
	/**** temporary variables **********************************************************************************************************************/
	VectorXd LS_XtY(ncov);
	MatrixXd LS_XtX(ncov, ncov);
	VectorXd theta0(ncov);
	VectorXd p(m); 
	VectorXd p0(m);		
	MatrixXd q(n_minus_n2, m);
	VectorXd q_row_sum(n_minus_n2);
	RowVectorXd q_col_sum(m);
	MatrixXd P_theta(n_minus_n2, m);
	double tol;
	int iter, idx, idx1;
	VectorXd X_uni_theta(m);
	VectorXd e_X_uni_theta(m);
	string error_message;
	double s0;
	VectorXd s1(ncov);
	MatrixXd s2(ncov, ncov);
	VectorXd lambda(n_event_uni);
	VectorXd lambda0(n_event_uni);
	VectorXd Lambda(n_event_uni);
	// /* RT test block */
	// time_t t1, t2;
	// /* RT test block */
	/**** temporary variables **********************************************************************************************************************/
	/*#############################################################################################################################################*/
	

	
	/*#############################################################################################################################################*/
	/**** EM algorithm *****************************************************************************************************************************/
	
	/**** parameter initialization *****************************************************************************************************************/
	theta.setZero();
	theta0.setZero();
		
	for (int k=0; k<m; k++)
	{
		p(k) = (X_uni_n(k)+0.)/(n2+0.);
	}
	p0 = p;
	
	flag_nonconvergence = false;
	flag_nonconvergence_cov = false;
	
	lambda.setZero();
	for (int i=0; i<n_event_uni; i++)
	{
		for (int i1=0; i1<n; i1++)
		{
			if (Y(i1) >= Y_uni_event(i))
			{
				lambda(i) ++;
			}
		}
		lambda(i) = (Y_uni_event_n(i)+0.)/lambda(i);
	}
	lambda0 = lambda;
	Lambda(0) = lambda(0);
	for (int i=1; i<n_event_uni; i++)
	{
		Lambda(i) = Lambda(i-1)+lambda(i);
	}
	/**** parameter initialization *****************************************************************************************************************/
	
	for (iter=0; iter<MAX_ITER; iter++) 
	{
		// /* RT test block */
		//	time(&t1);
		//	/* RT test block */
		
		/**** E-step *******************************************************************************************************************************/
		
		/**** update P_theta ***********************************************************************************************************************/
		X_uni_theta = X_uni*theta;
		e_X_uni_theta = e_X_uni_theta.array().exp();
		
		for (int i=0; i<n_minus_n2; i++)
		{
			idx = i+n2;
			if (Y_risk_ind(idx) > -1)
			{
				P_theta.row(i) = -Lambda(Y_risk_ind(idx))*e_X_uni_theta.transpose();
				P_theta.row(i) = P_theta.row(i).array().exp();
				if (Delta(idx) == 1)
				{
					P_theta.row(i) *= lambda(Y_risk_ind(idx));
					P_theta.row(i) = P_theta.row(i).array()*e_X_uni_theta.transpose().array();
				}				
			}
			else
			{
				if (Delta(idx) == 1)
				{
					stdError("Error: In TwoPhase_MLE0_noZW_coxph, the calculation of unique event times is wrong!");
				}
				
				P_theta.row(i).setOnes();			
			}
		}
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
		
		/**** update theta *************************************************************************************************************************/		
		LS_XtX.setZero();
		LS_XtY.setZero();
		
		for (int i=0; i<n2; i++)
		{
			if (Delta(i) == 1)
			{
				s0 = 0.;
				s1.setZero();
				s2.setZero();
				
				for (int i1=0; i1<n2; i1++)
				{
					if (Y(i1) >= Y(i))
					{
						s0 += e_X_uni_theta(X_uni_ind(i1));						
						s1.noalias() += X.row(i1).transpose()*e_X_uni_theta(X_uni_ind(i1));
						s2.noalias() += X.row(i1).transpose()*X.row(i1)*e_X_uni_theta(X_uni_ind(i1));
					}
				}
				
				for (int i1=0; i1<n_minus_n2; i1++)
				{
					idx1 = i1+n2;
					if (Y(idx1) >= Y(i))
					{
						for (int k=0; k<m; k++)
						{
							s0 += q(i1,k)*e_X_uni_theta(k);							
							s1.noalias() += q(i1,k)*X_uni.row(k).transpose()*e_X_uni_theta(k);
							s2.noalias() += q(i1,k)*X_uni.row(k).transpose()*X_uni.row(k)*e_X_uni_theta(k);							
						}
					}					
				}
				LS_XtY.noalias() += X.row(i).transpose()-s1/s0;
				LS_XtX.noalias() += s2/s0-s1*s1.transpose()/(s0*s0);
			}
		}
		
		for (int i=0; i<n_minus_n2; i++)
		{
			idx = i+n2;
			if (Delta(idx) == 1)
			{
				s0 = 0.;
				s1.setZero();
				s2.setZero();
				
				for (int i1=0; i1<n2; i1++)
				{
					if (Y(i1) >= Y(idx))
					{
						s0 += e_X_uni_theta(X_uni_ind(i1));						
						s1.noalias() += X.row(i1).transpose()*e_X_uni_theta(X_uni_ind(i1));
						s2.noalias() += X.row(i1).transpose()*X.row(i1)*e_X_uni_theta(X_uni_ind(i1));
					}
				}
				
				for (int i1=0; i1<n_minus_n2; i1++)
				{
					idx1 = i1+n2;
					if (Y(idx1) >= Y(idx))
					{
						for (int k=0; k<m; k++)
						{
							s0 += q(i1,k)*e_X_uni_theta(k);
							s1.noalias() += q(i1,k)*X_uni.row(k).transpose()*e_X_uni_theta(k);
							s2.noalias() += q(i1,k)*X_uni.row(k).transpose()*X_uni.row(k)*e_X_uni_theta(k);							
						}
					}					
				}
				
				for (int k=0; k<m; k++)
				{
					LS_XtY.noalias() += q(i,k)*X_uni.row(k).transpose();				
				}
				LS_XtY.noalias() -= s1/s0;			
				LS_XtX.noalias() += s2/s0-s1*s1.transpose()/(s0*s0);
			}
		}
		
		theta = LS_XtX.selfadjointView<Eigen::Upper>().ldlt().solve(LS_XtY);
		theta += theta0;
		/**** update theta *************************************************************************************************************************/
		
		/**** update lambda ************************************************************************************************************************/
		lambda.setZero();
		
		for (int i=0; i<n_event_uni; i++)
		{
			for (int i1=0; i1<n2; i1++)
			{
				if (Y(i1) >= Y_uni_event(i))
				{
					lambda(i) += e_X_uni_theta(X_uni_ind(i1));
				}
			}
			for (int i1=0; i1<n_minus_n2; i1++)
			{
				if (Y(i1+n2) >= Y_uni_event(i))
				{
					for (int k=0; k<m; k++)
					{
						lambda(i) += q(i1,k)*e_X_uni_theta(i1,k);
					}						
				}
			}
			lambda(i) = (Y_uni_event_n(i)+0.)/lambda(i);
		}

		Lambda(0) = lambda(0);
		for (int i=1; i<n_event_uni; i++)
		{
			Lambda(i) = Lambda(i-1)+lambda(i);
		}		
		/**** update lambda ************************************************************************************************************************/
		
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
		tol += (lambda-lambda0).array().abs().sum();
		tol += (p-p0).array().abs().sum();
		/**** calculate the sum of absolute differences between estimates in the current and previous iterations ***********************************/		
				
		/**** update parameters ********************************************************************************************************************/
		theta0 = theta;
		lambda0 = lambda;
		p0 = p;
		/**** update parameters ********************************************************************************************************************/
		
		/**** check convergence ********************************************************************************************************************/
		if (tol < TOL) 
		{
			break;
		}
		/**** check convergence ********************************************************************************************************************/
		
		// /* RT test block */
		//	time(&t2);
		//	Rcout << iter << '\t' << difftime(t2, t1) << '\t' << tol << endl;
		// /* RT test block */
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
		cov_theta.setConstant(-999.);
	} 
	else if (noSE) 
	{
		flag_nonconvergence_cov = true;
		cov_theta.setConstant(-999.);
	} 
	else 
	{
		// to be added
	}
	/**** variance estimation **********************************************************************************************************************/
	/*#############################################################################################################################################*/
	
	
	
	/*#############################################################################################################################################*/
	/**** return output to R ***********************************************************************************************************************/
	return List::create(Named("theta") = theta,
						Named("cov_theta") = cov_theta,
						Named("flag_nonconvergence") = flag_nonconvergence,
						Named("flag_nonconvergence_cov") = flag_nonconvergence_cov);
	/**** return output to R ***********************************************************************************************************************/
	/*#############################################################################################################################################*/
} // TwoPhase_MLE0_noZW_coxph

// double WaldCoxphGeneralSplineProfileLeftTrunc (Ref<MatrixXd> pB, Ref<RowVectorXd> p_col_sum, Ref<VectorXd> ZW_theta, Ref<VectorXd> X_uni_theta, Ref<MatrixXd> e_X_uni_theta,
	// Ref<VectorXd> e_X_theta, Ref<VectorXd> lambda, Ref<VectorXd> lambda0, Ref<VectorXd> Lambda,
	// Ref<VectorXd> q_row_sum, Ref<MatrixXd> p, Ref<MatrixXd> p0, Ref<MatrixXd> P_theta, Ref<MatrixXd> q, Ref<MatrixXd> logp,
	// const Ref<const VectorXd>& theta, const Ref<const VectorXd>& Y, const Ref<const VectorXd>& L, const Ref<const VectorXi>& Delta, const Ref<const MatrixXd>& X, 
	// const Ref<const MatrixXd>& Bspline_uni, const Ref<const MatrixXd>& ZW, const Ref<const MatrixXd>& X_uni, const Ref<const VectorXi>& X_uni_ind, 
	// const Ref<const VectorXi>& Bspline_uni_ind, const Ref<const VectorXd>& Y_uni_event, const Ref<const VectorXi>& Y_uni_event_n, 
	// const Ref<const VectorXi>& Y_risk_ind, const Ref<const VectorXi>& L_risk_ind, const Ref<const MatrixXd>& p_static, const int n, const int n2, const int m, const int n_event_uni, 
	// const int s, const int n_minus_n2, const int X_nc, const int ZW_nc, const int MAX_ITER, const double TOL) 
// {
	// /*#############################################################################################################################################*/
	// /**** temporary variables **********************************************************************************************************************/
	// double tol;
	// int iter, idx, idx1;
	// // /* RT test block */
	// // time_t t1, t2;
	// // /* RT test block */
	// /**** temporary variables **********************************************************************************************************************/
	// /*#############################################################################################################################################*/
		
	// /*#############################################################################################################################################*/
	// /**** EM algorithm *****************************************************************************************************************************/
	
	// /**** fixed quantities *************************************************************************************************************************/
	// ZW_theta = ZW*theta.tail(ZW_nc);
	// X_uni_theta = X_uni*theta.head(X_nc);
	// for (int i=0; i<n_minus_n2; i++)
	// {
		// for (int k=0; k<m; k++)
		// {
			// e_X_uni_theta(i,k) = ZW_theta(i+n2)+X_uni_theta(k);
		// }
	// }
	// e_X_uni_theta = e_X_uni_theta.array().exp();
	
	// for (int i=0; i<n2; i++)
	// {
		// e_X_theta(i) = ZW_theta(i)+X_uni_theta(X_uni_ind(i));
	// }
	// e_X_theta = e_X_theta.array().exp();
	// /**** fixed quantities *************************************************************************************************************************/
	
	// /**** parameter initialization *****************************************************************************************************************/
	// p_col_sum = p_static.colwise().sum();
	// for (int j=0; j<s; j++) 
	// {
		// p.col(j) = p_static.col(j)/p_col_sum(j);
	// }
	// p0 = p;
		
	// lambda.setZero();
	// for (int i=0; i<n_event_uni; i++)
	// {
		// for (int i1=0; i1<n; i1++)
		// {
			// if (Y(i1) >= Y_uni_event(i) && L(i1) <= Y_uni_event(i))
			// {
				// lambda(i) ++;
			// }
		// }
		// lambda(i) = (Y_uni_event_n(i)+0.)/lambda(i);
	// }
	// lambda0 = lambda;
	// Lambda(0) = lambda(0);
	// for (int i=1; i<n_event_uni; i++)
	// {
		// Lambda(i) = Lambda(i-1)+lambda(i);
	// }
	// /**** parameter initialization *****************************************************************************************************************/
	
	// for (iter=0; iter<MAX_ITER; iter++) 
	// {
		// // /* RT test block */
		// // time(&t1);
		// // /* RT test block */
		
		// /**** E-step *******************************************************************************************************************************/
		
		// /**** update pB ****************************************************************************************************************************/
		// pB = Bspline_uni*p.transpose();
		// /**** update pB ****************************************************************************************************************************/
				
		// /**** update P_theta ***********************************************************************************************************************/		
		// for (int i=0; i<n_minus_n2; i++)
		// {
			// idx = i+n2;
			// if (Y_risk_ind(idx) > -1)
			// {
				// if (L_risk_ind(idx) == -1)
				// {
					// P_theta.row(i) = -Lambda(Y_risk_ind(idx))*e_X_uni_theta.row(i);
				// }
				// else
				// {
					// P_theta.row(i) = (-Lambda(Y_risk_ind(idx))+Lambda(L_risk_ind(idx)))*e_X_uni_theta.row(i);
				// }
				// P_theta.row(i) = P_theta.row(i).array().exp();
				// if (Delta(idx) == 1)
				// {
					// P_theta.row(i) *= lambda(Y_risk_ind(idx));
					// P_theta.row(i) = P_theta.row(i).array()*e_X_uni_theta.row(i).array();
				// }
			// }
			// else
			// {
				// if (Delta(idx) == 1)
				// {
					// stdError("Error: In WaldCoxphGeneralSplineProfileLeftTrunc, the calculation of unique event times is wrong!");
				// }
				
				// P_theta.row(i).setOnes();			
			// }
		// }
		// /**** update P_theta ***********************************************************************************************************************/
		
		// /**** update q, q_row_sum ******************************************************************************************************************/
		// for (int i=0; i<n_minus_n2; i++) 
		// {
			// for (int k=0; k<m; k++) 
			// {
				// q(i,k) = P_theta(i,k)*pB(Bspline_uni_ind(i+n2),k);
			// }
		// }
		// q_row_sum = q.rowwise().sum();		
		// for (int i=0; i<n_minus_n2; i++) 
		// {
			// q.row(i) /= q_row_sum(i);
		// }
		// /**** update q, q_row_sum ******************************************************************************************************************/
		
		// /**** E-step *******************************************************************************************************************************/
		
		
		// /**** M-step *******************************************************************************************************************************/
		
		// /**** update lambda ************************************************************************************************************************/
		// lambda.setZero();
		
		// for (int i=0; i<n_event_uni; i++)
		// {
			// for (int i1=0; i1<n2; i1++)
			// {
				// if (Y(i1) >= Y_uni_event(i) && L(i1) <= Y_uni_event(i))
				// {
					// lambda(i) += e_X_theta(i1);
				// }
			// }
			// for (int i1=0; i1<n_minus_n2; i1++)
			// {
				// idx1 = i1+n2;
				// if (Y(idx1) >= Y_uni_event(i) && L(idx1) <= Y_uni_event(i))
				// {
					// for (int k=0; k<m; k++)
					// {
						// lambda(i) += q(i1,k)*e_X_uni_theta(i1,k);
					// }						
				// }
			// }
			// lambda(i) = (Y_uni_event_n(i)+0.)/lambda(i);
		// }
		// Lambda(0) = lambda(0);
		// for (int i=1; i<n_event_uni; i++)
		// {
			// Lambda(i) = Lambda(i-1)+lambda(i);
		// }		
		// /**** update lambda ************************************************************************************************************************/
		
		// /**** update p *****************************************************************************************************************************/
		// p.setZero();		
		// for (int i=0; i<n_minus_n2; i++) 
		// {
			// for (int k=0; k<m; k++) 
			// {
				// for (int j=0; j<s; j++) 
				// {
					// p(k,j) += Bspline_uni(Bspline_uni_ind(i+n2),j)*P_theta(i,k)/q_row_sum(i);
				// }
			// }
		// }		
		// p = p.array()*p0.array();
		// p += p_static;
		// p_col_sum = p.colwise().sum();
		// for (int j=0; j<s; j++) 
		// {
			// p.col(j) /= p_col_sum(j);
		// }
		// /**** update p *****************************************************************************************************************************/
		
		// /**** M-step *******************************************************************************************************************************/
		
		
		// /**** calculate the sum of absolute differences between estimates in the current and previous iterations ***********************************/
		// tol = (lambda-lambda0).array().abs().sum();
		// tol += (p-p0).array().abs().sum();
		// /**** calculate the sum of absolute differences between estimates in the current and previous iterations ***********************************/		
				
		// /**** update parameters ********************************************************************************************************************/
		// lambda0 = lambda;
		// p0 = p;
		// /**** update parameters ********************************************************************************************************************/
		
		// /**** check convergence ********************************************************************************************************************/
		// if (tol < TOL) 
		// {
			// break;
		// }
		// /**** check convergence ********************************************************************************************************************/

		// // /* RT test block */
		// // time(&t2);
		// // Rcout << iter << '\t' << difftime(t2, t1) << '\t' << tol << endl;
		// // /* RT test block */
	// }
	// /**** EM algorithm *****************************************************************************************************************************/
	// /*#############################################################################################################################################*/

	// if (iter == MAX_ITER) 
	// {
		// return -999.;
	// } 
	// else 
	// {
		// /**** calculate the likelihood *************************************************************************************************************/
		// double loglik;
		
		// logp = p.array().log();
		// for (int k=0; k<m; k++)
		// {
			// for (int j=0; j<s; j++)
			// {
				// if (p(k,j) <= 0.)
				// {
					// logp(k,j) = 0.;
				// }
			// }
		// }
		// pB = Bspline_uni*logp.transpose();
		
		// loglik = 0.;
		// for (int i=0; i<n2; i++) {
			// loglik += pB(Bspline_uni_ind(i),X_uni_ind(i));
			// if (L_risk_ind(i) == -1)
			// {
				// loglik -= Lambda(Y_risk_ind(i))*e_X_theta(i);
			// }
			// else
			// {
				// loglik -= (Lambda(Y_risk_ind(i))-Lambda(L_risk_ind(i)))*e_X_theta(i);
			// }
			// if (Delta(i) == 1)
			// {
				// loglik += log(lambda(Y_risk_ind(i)))+ZW_theta(i)+X_uni_theta(X_uni_ind(i));
			// }
		// }
		
		// pB = Bspline_uni*p.transpose();;
		
		// for (int i=0; i<n_minus_n2; i++) 
		// {
			// for (int k=0; k<m; k++) 
			// {
				// q(i,k) = P_theta(i,k)*pB(Bspline_uni_ind(i+n2),k);
			// }
		// }
		// q_row_sum = q.rowwise().sum();		
		
		// loglik += q_row_sum.array().log().sum();	
		// /**** calculate the likelihood *************************************************************************************************************/
		
		// return loglik;	
	// }
// } // WaldCoxphGeneralSplineProfileLeftTrunc

// RcppExport SEXP TwoPhase_GeneralSpline_coxph_LeftTrunc (SEXP Y_R, SEXP L_R, SEXP Delta_R, SEXP X_R, SEXP ZW_R, SEXP Bspline_R, SEXP hn_R, SEXP MAX_ITER_R, SEXP TOL_R, SEXP noSE_R) 
// {
	// /*#############################################################################################################################################*/
	// /**** pass arguments from R to cpp *************************************************************************************************************/
	// const MapVecd Y(as<MapVecd>(Y_R));
	// const MapVecd L(as<MapVecd>(L_R));
	// const MapVeci Delta(as<MapVeci>(Delta_R));
	// const MapMatd X(as<MapMatd>(X_R));
	// const MapMatd ZW(as<MapMatd>(ZW_R));
	// const MapMatd Bspline(as<MapMatd>(Bspline_R));
	// const double hn = NumericVector(hn_R)[0];
	// const int MAX_ITER = IntegerVector(MAX_ITER_R)[0];
	// const double TOL = NumericVector(TOL_R)[0];
	// const int noSE = IntegerVector(noSE_R)[0];
	// /**** pass arguments from R to cpp *************************************************************************************************************/
	// /*#############################################################################################################################################*/
	
	
	
	// /*#############################################################################################################################################*/
	// /**** some useful constants ********************************************************************************************************************/
	// const int n = Y.size();  // number of subjects in the first phase
	// const int n2 = X.rows(); // number of subjects in the second phase
	// const int n_minus_n2 = n-n2; // number of subjects not selected in the second phase
	// const int ZW_nc = ZW.cols(); // number of inexpensive covariates
	// const int X_nc = X.cols(); // number of expensive covariates X
	// const int ncov = X_nc+ZW_nc; // number of all covariates
	// const int s = Bspline.cols(); // number of B-spline functions
	// /**** some useful constants ********************************************************************************************************************/
	// /*#############################################################################################################################################*/	
	

	
	// /*#############################################################################################################################################*/	
	// /**** summarize observed distinct rows of X ****************************************************************************************************/
	// // m: number of distinct rows of X
	// // X_uni_ind(n2): row index of rows of X in X_uni
	// // X_uni(m, X_nc): distinct rows of X
	// // X_uni_n(m): count of appearances of each distinct row of X	
	// int m;
	// VectorXi X_index(n2);
	// VectorXi X_uni_ind(n2);	
	// indexx_Matrix_Row(X, X_index);
	// Num_Uni_Matrix_Row(X, X_index, m);	
	// MatrixXd X_uni(m, X_nc); 
	// VectorXi X_uni_n(m);	
	// Create_Uni_Matrix_Row(X, X_index, X_uni, X_uni_ind, X_uni_n);
	// /**** summarize observed distinct rows of X ****************************************************************************************************/
	// /*#############################################################################################################################################*/
	

	
	// /*#############################################################################################################################################*/
	// /**** summarize observed distinct rows of Bspline **********************************************************************************************/
	// // m_B: number of distinct rows of Bspline
	// // Bspline_uni_ind(n): row index of rows of Bspline in Bspline_uni
	// // Bspline_uni(m_B, s): distinct rows of Bspline
	// // Bspline_uni_n(m_B): count of appearances of each distinct row of Bspline
	// int m_B;
	// VectorXi Bspline_index(n);
	// VectorXi Bspline_uni_ind(n);	
	// indexx_Matrix_Row(Bspline, Bspline_index);
	// Num_Uni_Matrix_Row(Bspline, Bspline_index, m_B);	
	// MatrixXd Bspline_uni(m_B, s);
	// VectorXi Bspline_uni_n(m_B);
	// Create_Uni_Matrix_Row(Bspline, Bspline_index, Bspline_uni, Bspline_uni_ind, Bspline_uni_n);
	// /**** summarize observed distinct rows of Bspline **********************************************************************************************/
	// /*#############################################################################################################################################*/
	
	
	
	// /*#############################################################################################################################################*/	
	// /**** summarize observed distinct values of Y **************************************************************************************************/
	// // n_event_uni: number of distinct values of Y where Delta=1
	// // Y_uni_event: vector of unique events
	// // Y_risk_ind: vector of indexes of which one of the risk sets each element in Y corresponds to.
	// // Y_uni_event_n: count of appearances of each distinct event time	
	// int n_event_uni;
	// VectorXi Y_index(n);	
	// indexx_Vector(Y, Y_index);
	// VectorXi L_index(n);
	// indexx_Vector(L, L_index);
	// Num_Distinct_Events(Y, Y_index, Delta, n_event_uni);
	// VectorXd Y_uni_event(n_event_uni); 
	// VectorXi Y_risk_ind(n);
	// VectorXi Y_uni_event_n(n_event_uni);
	// VectorXi L_risk_ind(n);
	// Create_Uni_Events_LeftTrunc (Y, L, Y_index, L_index, Delta, Y_uni_event, Y_risk_ind, Y_uni_event_n, L_risk_ind);
	// /**** summarize observed distinct values of Y **************************************************************************************************/
	// /*#############################################################################################################################################*/
	
	
	
	// /*#############################################################################################################################################*/
	// /**** some fixed quantities in the EM algorithm ************************************************************************************************/		
	// // p
	// MatrixXd p_static(m, s); 
	// p_static.setZero();
	// for (int i=0; i<n2; i++) 
	// {
		// p_static.row(X_uni_ind(i)) += Bspline.row(i);
	// }	
	// /**** some fixed quantities in the EM algorithm ************************************************************************************************/
	// /*#############################################################################################################################################*/
	

	
	// /*#############################################################################################################################################*/
	// /**** output ***********************************************************************************************************************************/
	// VectorXd theta(ncov); // regression coefficients
	// MatrixXd cov_theta(ncov, ncov); // covariance matrix
	// bool flag_nonconvergence; // flag of none convergence in the estimation of regression coefficients
	// bool flag_nonconvergence_cov; // flag of none convergence in the estimation of covariance matrix
	// /**** output ***********************************************************************************************************************************/
	// /*#############################################################################################################################################*/
	

		
	// /*#############################################################################################################################################*/
	// /**** temporary variables **********************************************************************************************************************/
	// VectorXd LS_XtY(ncov);
	// MatrixXd LS_XtX(ncov, ncov);
	// VectorXd theta0(ncov);
	// MatrixXd p(m, s); 
	// MatrixXd p0(m, s);
	// RowVectorXd p_col_sum(s);	
	// MatrixXd q(n_minus_n2, m);
	// VectorXd q_row_sum(n_minus_n2);
	// MatrixXd pB(m_B, m);
	// MatrixXd P_theta(n_minus_n2, m);
	// double tol;
	// int iter, idx, idx1;
	// VectorXd ZW_theta(n);
	// VectorXd X_uni_theta(m);
	// MatrixXd e_X_uni_theta(n_minus_n2, m);
	// VectorXd e_X_theta(n2);
	// string error_message;
	// double s0;
	// VectorXd s1(ncov);
	// MatrixXd s2(ncov, ncov);
	// VectorXd lambda(n_event_uni);
	// VectorXd lambda0(n_event_uni);
	// VectorXd Lambda(n_event_uni);
	// // /* RT test block */
	// // time_t t1, t2;
	// // /* RT test block */
	// /**** temporary variables **********************************************************************************************************************/
	// /*#############################################################################################################################################*/
	

	
	// /*#############################################################################################################################################*/
	// /**** EM algorithm *****************************************************************************************************************************/
	
	// /**** parameter initialization *****************************************************************************************************************/
	// theta.setZero();
	// theta0.setZero();
	
	// p_col_sum = p_static.colwise().sum();
	// for (int j=0; j<s; j++) 
	// {
		// p.col(j) = p_static.col(j)/p_col_sum(j);
	// }
	// p0 = p;
	
	// flag_nonconvergence = false;
	// flag_nonconvergence_cov = false;
	
	// lambda.setZero();
	// for (int i=0; i<n_event_uni; i++)
	// {
		// for (int i1=0; i1<n; i1++)
		// {
			// if (Y(i1) >= Y_uni_event(i) && L(i1) <= Y_uni_event(i))
			// {
				// lambda(i) ++;
			// }
		// }
		// lambda(i) = (Y_uni_event_n(i)+0.)/lambda(i);
	// }
	// lambda0 = lambda;
	// Lambda(0) = lambda(0);
	// for (int i=1; i<n_event_uni; i++)
	// {
		// Lambda(i) = Lambda(i-1)+lambda(i);
	// }
	// /**** parameter initialization *****************************************************************************************************************/
	
	// for (iter=0; iter<MAX_ITER; iter++) 
	// {
		// // /* RT test block */
		// // time(&t1);
		// // /* RT test block */
		
		// /**** E-step *******************************************************************************************************************************/
		
		// /**** update pB ****************************************************************************************************************************/
		// pB = Bspline_uni*p.transpose();
		// /**** update pB ****************************************************************************************************************************/
				
		// /**** update P_theta ***********************************************************************************************************************/
		// ZW_theta = ZW*theta.tail(ZW_nc);
		// X_uni_theta = X_uni*theta.head(X_nc);
		// for (int i=0; i<n_minus_n2; i++)
		// {
			// for (int k=0; k<m; k++)
			// {
				// e_X_uni_theta(i,k) = ZW_theta(i+n2)+X_uni_theta(k);
			// }
		// }
		// e_X_uni_theta = e_X_uni_theta.array().exp();
		
		// for (int i=0; i<n_minus_n2; i++)
		// {
			// idx = i+n2;
			// if (Y_risk_ind(idx) > -1)
			// {
				// if (L_risk_ind(idx) == -1)
				// {
					// P_theta.row(i) = -Lambda(Y_risk_ind(idx))*e_X_uni_theta.row(i);
				// }
				// else
				// {
					// P_theta.row(i) = (-Lambda(Y_risk_ind(idx))+Lambda(L_risk_ind(idx)))*e_X_uni_theta.row(i);
				// }				
				// P_theta.row(i) = P_theta.row(i).array().exp();
				// if (Delta(idx) == 1)
				// {
					// P_theta.row(i) *= lambda(Y_risk_ind(idx));
					// P_theta.row(i) = P_theta.row(i).array()*e_X_uni_theta.row(i).array();
				// }
			// }
			// else
			// {
				// if (Delta(idx) == 1)
				// {
					// stdError("Error: In TwoPhase_GeneralSpline_coxph_LeftTrunc, the calculation of unique event times is wrong!");
				// }
				
				// P_theta.row(i).setOnes();			
			// }
		// }
		// /**** update P_theta ***********************************************************************************************************************/
		
		// /**** update q, q_row_sum ******************************************************************************************************************/
		// for (int i=0; i<n_minus_n2; i++) 
		// {
			// for (int k=0; k<m; k++) 
			// {
				// q(i,k) = P_theta(i,k)*pB(Bspline_uni_ind(i+n2),k);
			// }
		// }
		// q_row_sum = q.rowwise().sum();		
		// for (int i=0; i<n_minus_n2; i++) 
		// {
			// q.row(i) /= q_row_sum(i);
		// }
		// /**** update q, q_row_sum ******************************************************************************************************************/
		
		// /**** E-step *******************************************************************************************************************************/
		
		
		// /**** M-step *******************************************************************************************************************************/
		
		// /**** update theta *************************************************************************************************************************/		
		// for (int i=0; i<n2; i++)
		// {
			// e_X_theta(i) = ZW_theta(i)+X_uni_theta(X_uni_ind(i));
		// }
		// e_X_theta = e_X_theta.array().exp();
		
		// LS_XtX.setZero();
		// LS_XtY.setZero();
		// for (int i=0; i<n2; i++)
		// {
			// if (Delta(i) == 1)
			// {
				// s0 = 0.;
				// s1.setZero();
				// s2.setZero();
				
				// for (int i1=0; i1<n2; i1++)
				// {
					// if (Y(i1) >= Y(i) && L(i1) <= Y(i))
					// {
						// s0 += e_X_theta(i1);
						// s1.head(X_nc).noalias() += X.row(i1).transpose()*e_X_theta(i1);
						// s1.tail(ZW_nc).noalias() += ZW.row(i1).transpose()*e_X_theta(i1);						
						// s2.topLeftCorner(X_nc,X_nc).noalias() += X.row(i1).transpose()*X.row(i1)*e_X_theta(i1);
						// s2.topRightCorner(X_nc,ZW_nc).noalias() += X.row(i1).transpose()*ZW.row(i1)*e_X_theta(i1);
						// s2.bottomRightCorner(ZW_nc,ZW_nc).noalias() += ZW.row(i1).transpose()*ZW.row(i1)*e_X_theta(i1);
					// }
				// }
				// for (int i1=0; i1<n_minus_n2; i1++)
				// {
					// idx1 = i1+n2;
					// if (Y(idx1) >= Y(i) && L(idx1) <= Y(i))
					// {
						// for (int k=0; k<m; k++)
						// {
							// s0 += q(i1,k)*e_X_uni_theta(i1,k);
							// s1.head(X_nc).noalias() += q(i1,k)*X_uni.row(k).transpose()*e_X_uni_theta(i1,k);
							// s1.tail(ZW_nc).noalias() += q(i1,k)*ZW.row(idx1).transpose()*e_X_uni_theta(i1,k);
							// s2.topLeftCorner(X_nc,X_nc).noalias() += q(i1,k)*X_uni.row(k).transpose()*X_uni.row(k)*e_X_uni_theta(i1,k);
							// s2.topRightCorner(X_nc,ZW_nc).noalias() += q(i1,k)*X_uni.row(k).transpose()*ZW.row(idx1)*e_X_uni_theta(i1,k);
							// s2.bottomRightCorner(ZW_nc,ZW_nc).noalias() += q(i1,k)*ZW.row(idx1).transpose()*ZW.row(idx1)*e_X_uni_theta(i1,k);
						// }
					// }					
				// }
				// LS_XtY.head(X_nc).noalias() += X.row(i).transpose()-s1.head(X_nc)/s0;
				// LS_XtY.tail(ZW_nc).noalias() += ZW.row(i).transpose()-s1.tail(ZW_nc)/s0;
				// LS_XtX.noalias() += s2/s0-s1*s1.transpose()/(s0*s0);
			// }
		// }

		// for (int i=0; i<n_minus_n2; i++)
		// {
			// idx = i+n2;
			// if (Delta(idx) == 1)
			// {
				// s0 = 0.;
				// s1.setZero();
				// s2.setZero();
				
				// for (int i1=0; i1<n2; i1++)
				// {
					// if (Y(i1) >= Y(idx) && L(i1) <= Y(idx))
					// {
						// s0 += e_X_theta(i1);
						// s1.head(X_nc).noalias() += X.row(i1).transpose()*e_X_theta(i1);
						// s1.tail(ZW_nc).noalias() += ZW.row(i1).transpose()*e_X_theta(i1);						
						// s2.topLeftCorner(X_nc,X_nc).noalias() += X.row(i1).transpose()*X.row(i1)*e_X_theta(i1);
						// s2.topRightCorner(X_nc,ZW_nc).noalias() += X.row(i1).transpose()*ZW.row(i1)*e_X_theta(i1);
						// s2.bottomRightCorner(ZW_nc,ZW_nc).noalias() += ZW.row(i1).transpose()*ZW.row(i1)*e_X_theta(i1);
					// }
				// }
				
				// for (int i1=0; i1<n_minus_n2; i1++)
				// {
					// idx1 = i1+n2;
					// if (Y(idx1) >= Y(idx) && L(idx1) <= Y(idx))
					// {
						// for (int k=0; k<m; k++)
						// {
							// s0 += q(i1,k)*e_X_uni_theta(i1,k);							
							// s1.head(X_nc).noalias() += q(i1,k)*X_uni.row(k).transpose()*e_X_uni_theta(i1,k);
							// s1.tail(ZW_nc).noalias() += q(i1,k)*ZW.row(idx1).transpose()*e_X_uni_theta(i1,k);						
							// s2.topLeftCorner(X_nc,X_nc).noalias() += q(i1,k)*X_uni.row(k).transpose()*X_uni.row(k)*e_X_uni_theta(i1,k);
							// s2.topRightCorner(X_nc,ZW_nc).noalias() += q(i1,k)*X_uni.row(k).transpose()*ZW.row(idx1)*e_X_uni_theta(i1,k);
							// s2.bottomRightCorner(ZW_nc,ZW_nc).noalias() += q(i1,k)*ZW.row(idx1).transpose()*ZW.row(idx1)*e_X_uni_theta(i1,k);							
						// }
					// }					
				// }				
				// for (int k=0; k<m; k++)
				// {
					// LS_XtY.head(X_nc).noalias() += q(i,k)*X_uni.row(k).transpose();				
				// }
				// LS_XtY.head(X_nc).noalias() -= s1.head(X_nc)/s0;			
				// LS_XtY.tail(ZW_nc).noalias() += ZW.row(idx).transpose()-s1.tail(ZW_nc)/s0;
				// LS_XtX.noalias() += s2/s0-s1*s1.transpose()/(s0*s0);
			// }
		// }

		// theta = LS_XtX.selfadjointView<Eigen::Upper>().ldlt().solve(LS_XtY);
		// theta += theta0;
		// /**** update theta *************************************************************************************************************************/
		
		// /**** update lambda ************************************************************************************************************************/
		// lambda.setZero();
		
		// for (int i=0; i<n_event_uni; i++)
		// {
			// for (int i1=0; i1<n2; i1++)
			// {
				// if (Y(i1) >= Y_uni_event(i) && L(i1) <= Y_uni_event(i))
				// {
					// lambda(i) += e_X_theta(i1);
				// }
			// }
			// for (int i1=0; i1<n_minus_n2; i1++)
			// {
				// idx1 = i1+n2;
				// if (Y(idx1) >= Y_uni_event(i) && L(idx1) <= Y_uni_event(i))
				// {
					// for (int k=0; k<m; k++)
					// {
						// lambda(i) += q(i1,k)*e_X_uni_theta(i1,k);
					// }						
				// }
			// }
			// lambda(i) = (Y_uni_event_n(i)+0.)/lambda(i);
		// }
		// Lambda(0) = lambda(0);
		// for (int i=1; i<n_event_uni; i++)
		// {
			// Lambda(i) = Lambda(i-1)+lambda(i);
		// }		
		// /**** update lambda ************************************************************************************************************************/
		
		// /**** update p *****************************************************************************************************************************/
		// p.setZero();		
		// for (int i=0; i<n_minus_n2; i++) 
		// {
			// for (int k=0; k<m; k++) 
			// {
				// for (int j=0; j<s; j++) 
				// {
					// p(k,j) += Bspline_uni(Bspline_uni_ind(i+n2),j)*P_theta(i,k)/q_row_sum(i);
				// }
			// }
		// }		
		// p = p.array()*p0.array();
		// p += p_static;
		// p_col_sum = p.colwise().sum();
		// for (int j=0; j<s; j++) 
		// {
			// p.col(j) /= p_col_sum(j);
		// }
		// /**** update p *****************************************************************************************************************************/
		
		// /**** M-step *******************************************************************************************************************************/
		
		
		// /**** calculate the sum of absolute differences between estimates in the current and previous iterations ***********************************/
		// tol = (theta-theta0).array().abs().sum();
		// tol += (lambda-lambda0).array().abs().sum();
		// tol += (p-p0).array().abs().sum();
		// /**** calculate the sum of absolute differences between estimates in the current and previous iterations ***********************************/		
				
		// /**** update parameters ********************************************************************************************************************/
		// theta0 = theta;
		// lambda0 = lambda;
		// p0 = p;
		// /**** update parameters ********************************************************************************************************************/
		
		// /**** check convergence ********************************************************************************************************************/
		// if (tol < TOL) 
		// {
			// break;
		// }
		// /**** check convergence ********************************************************************************************************************/

		// // /* RT test block */
		// // time(&t2);
		// // Rcout << iter << '\t' << difftime(t2, t1) << '\t' << tol << endl;
		// // /* RT test block */
	// }
	// /**** EM algorithm *****************************************************************************************************************************/
	// /*#############################################################################################################################################*/

	
	
	// /*#############################################################################################################################################*/
	// /**** variance estimation **********************************************************************************************************************/
	// if (iter == MAX_ITER) 
	// {
		// flag_nonconvergence = true;
		// flag_nonconvergence_cov = true;
		// theta.setConstant(-999.);
		// cov_theta.setConstant(-999.);
	// } 
	// else if (noSE) 
	// {
		// flag_nonconvergence_cov = true;
		// cov_theta.setConstant(-999.);
	// } 
	// else 
	// {
		// VectorXd profile_vec(ncov);
		// MatrixXd profile_mat(ncov, ncov);
		// MatrixXd logp(m, s);
		// MatrixXd inv_profile_mat(ncov, ncov);
		// double loglik;
				
		// profile_mat.setZero();
		// profile_vec.setZero();
		
		// loglik = WaldCoxphGeneralSplineProfileLeftTrunc(pB, p_col_sum, ZW_theta, X_uni_theta, e_X_uni_theta, e_X_theta, lambda, lambda0, Lambda, q_row_sum, p, p0, P_theta, q, logp,
			// theta, Y, L, Delta, X, Bspline_uni, ZW, X_uni, X_uni_ind, Bspline_uni_ind, Y_uni_event, Y_uni_event_n, 
			// Y_risk_ind, L_risk_ind, p_static, n, n2, m, n_event_uni, s, n_minus_n2, X_nc, ZW_nc, MAX_ITER, TOL);

		// if (loglik == -999.) 
		// {
			// flag_nonconvergence_cov = true;
		// }
		// for (int i=0; i<ncov; i++) 
		// {
			// for (int j=i; j<ncov; j++) 
			// {
				// profile_mat(i,j) = loglik;
			// }
		// }

		// for (int i=0; i<ncov; i++) 
		// {
			// theta0 = theta;
			// theta0(i) += hn;
			// profile_vec(i) = WaldCoxphGeneralSplineProfileLeftTrunc(pB, p_col_sum, ZW_theta, X_uni_theta, e_X_uni_theta, e_X_theta, lambda, lambda0, Lambda, q_row_sum, p, p0, P_theta, q, logp,
				// theta0, Y, L, Delta, X, Bspline_uni, ZW, X_uni, X_uni_ind, Bspline_uni_ind, Y_uni_event, Y_uni_event_n, 
				// Y_risk_ind, L_risk_ind, p_static, n, n2, m, n_event_uni, s, n_minus_n2, X_nc, ZW_nc, MAX_ITER, TOL);
		// }
		// for (int i=0; i<ncov; i++) 
		// {
			// if(profile_vec(i) == -999.) 
			// {
				// flag_nonconvergence_cov = true;
			// }
		// }
		
		// for (int i=0; i<ncov; i++) 
		// {
			// for (int j=i; j<ncov; j++) 
			// {
				// theta0 = theta;
				// theta0(i) += hn;
				// theta0(j) += hn;
				// loglik = WaldCoxphGeneralSplineProfileLeftTrunc(pB, p_col_sum, ZW_theta, X_uni_theta, e_X_uni_theta, e_X_theta, lambda, lambda0, Lambda, q_row_sum, p, p0, P_theta, q, logp,
					// theta0, Y, L, Delta, X, Bspline_uni, ZW, X_uni, X_uni_ind, Bspline_uni_ind, Y_uni_event, Y_uni_event_n, 
					// Y_risk_ind, L_risk_ind, p_static, n, n2, m, n_event_uni, s, n_minus_n2, X_nc, ZW_nc, MAX_ITER, TOL);
				// if (loglik == -999.) 
				// {
					// flag_nonconvergence_cov = true;
				// }				
				// profile_mat(i,j) += loglik;
				// profile_mat(i,j) -= profile_vec(i)+profile_vec(j);
			// }
		// }
				
		// if (flag_nonconvergence_cov == true) 
		// {
			// cov_theta.setConstant(-999.);
		// } 
		// else 
		// {		
			// for (int i=0; i<ncov; i++) 
			// {
				// for (int j=i+1; j<ncov; j++) 
				// {
					// profile_mat(j,i) = profile_mat(i,j);
				// }
			// }		
			// profile_mat /= hn*hn;
			// profile_mat = -profile_mat;

			// inv_profile_mat = profile_mat.selfadjointView<Eigen::Upper>().ldlt().solve(MatrixXd::Identity(ncov, ncov));
			// cov_theta = inv_profile_mat.topLeftCorner(ncov,ncov);
		// }
	// }
	// /**** variance estimation **********************************************************************************************************************/
	// /*#############################################################################################################################################*/
	
	
	
	// /*#############################################################################################################################################*/
	// /**** return output to R ***********************************************************************************************************************/
	// return List::create(Named("theta") = theta,
						// Named("cov_theta") = cov_theta,
						// Named("flag_nonconvergence") = flag_nonconvergence,
						// Named("flag_nonconvergence_cov") = flag_nonconvergence_cov);
	// /**** return output to R ***********************************************************************************************************************/
	// /*#############################################################################################################################################*/
// } // TwoPhase_GeneralSpline_coxph_LeftTrunc
