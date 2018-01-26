#include <string>
#include <iostream>
#include <ctime>
#include <cmath>
#include "utility.h"

#define NSTACK 50 // indexx, indexx_Matrix_Row
#define M 7 // indexx, indexx_Matrix_Row
#define M_PI 3.14159265358979323846 // Wald_Linear_Kernel_Loglike

void stdError (const string reason) 
{
	Rcpp::Rcout << reason << endl;
	Rcpp::stop("Program was stopped due to error(s) listed above.\n");
} // stdError

bool BigArray (const Ref<const RowVectorXd>& arr1, const Ref<const RowVectorXd>& arr2, int p) 
{	
	int i = 0;

	while (arr1(i) == arr2(i)) 
	{
		if (i == p-1) return false;
		i++;
	}

	return arr1(i)>arr2(i) ? true:false;
} // BigArray

bool SmallArray (const Ref<const RowVectorXd>& arr1, const Ref<const RowVectorXd>& arr2, int p)
{	
	int i = 0;

	while (arr1(i) == arr2(i)) 
	{
		if (i == p-1) return false;
		i++;
	}

	return arr1(i)<arr2(i) ? true:false;
} // SmallArray

bool EqualArray (const Ref<const RowVectorXd>& arr1, const Ref<const RowVectorXd>& arr2, int p)
{	
	int i = 0;

	while (arr1(i) == arr2(i)) 
	{
		if(i == p-1) return true;
		i++;
	}
	
	return false;
} // EqualArray

void indexx_Matrix_Row (const Ref<const MatrixXd>& mat, Ref<VectorXi> indx) 
{	
	int n = mat.rows(), ncol = mat.cols(), i, indxt, ir=n-1, j, k, l=0, jstack=0, tmp;
	RowVectorXd a(ncol);
	VectorXi istack(NSTACK+1);

	for (j=0; j<n; j++) indx(j) = j;
	for (;;) 
	{
		if (ir-l < M) 
		{
			for (j=l+1; j<=ir; j++) 
			{
				indxt = indx(j);
				a = mat.row(indxt);
				for (i=j-1; i>=l; i--) 
				{
					if (!SmallArray(a, mat.row(indx(i)), ncol)) break;
					indx(i+1) = indx(i);
				}
				indx(i+1) = indxt;
			}
			if (jstack == 0) break;
			ir = istack(jstack--);
			l = istack(jstack--);
		} 
		else 
		{
			k = (l+ir) >> 1;
			tmp = indx(k);
			indx(k) = indx(l+1);
			indx(l+1) = tmp;
			if (BigArray(mat.row(indx(l)), mat.row(indx(ir)), ncol)) 
			{
				tmp = indx(l);
				indx(l) = indx(ir);
				indx(ir) = tmp;
			}
			if (BigArray(mat.row(indx(l+1)), mat.row(indx(ir)), ncol)) 
			{
				tmp = indx(l+1);
				indx(l+1) = indx(ir);
				indx(ir) = tmp;
			}
			if (BigArray(mat.row(indx(l)), mat.row(indx(l+1)), ncol)) 
			{
				tmp = indx(l);
				indx(l) = indx(l+1);
				indx(l+1) = tmp;
			}
			i = l+1;
			j = ir;
			indxt = indx(l+1);
			a = mat.row(indxt);
			for (;;) 
			{
				do i++; while (SmallArray(mat.row(indx(i)), a, ncol));
				do j--; while (BigArray(mat.row(indx(j)), a, ncol));
				if (j < i) break;
				tmp = indx(i);
				indx(i) = indx(j);
				indx(j) = tmp;
			}
			indx(l+1) = indx(j);
			indx(j) = indxt;
			jstack += 2;
			if (jstack > NSTACK) stdError("Error: NSTACK too small in indexx_Matrix_Row!");
			if (ir-i+1 >= j-l) 
			{
				istack(jstack) = ir;
				istack(jstack-1) = i;
				ir = j-1;
			} 
			else 
			{
				istack(jstack) = j-1;
				istack(jstack-1) = l;
				l = i;
			}
		}
	}
} // indexx_Matrix_Row

void Num_Uni_Matrix_Row (const Ref<const MatrixXd>& mat, const Ref<const VectorXi>& index, int& n1) 
{
/***************************************************************************************
 Find the number of unique rows in "mat" and return it as "n1". "index" is the index 
 vector of the rows of "mat" output by "indexx_Matrix_Row". "mat" and "index" are not 
 changed.
***************************************************************************************/	
	int ncol, nrow;
	
	ncol = mat.cols();
	nrow = mat.rows();
	
	n1 = 1;
	for(int i=0; i<nrow-1; i++) 
	{
		if(EqualArray(mat.row(index(i)), mat.row(index(i+1)), ncol));
		else n1++; 
	}
} // Num_Uni_Matrix_Row

void Create_Uni_Matrix_Row (const Ref<const MatrixXd>& mat, const Ref<const VectorXi>& index, Ref<MatrixXd> uni_mat,
	Ref<VectorXi> ind, Ref<VectorXi> N_uni) 
{
/***************************************************************************************
 Create unique covariates. "mat" is a matrix (not changed). "index" is the index vector 
 of the rows of mat output by indexx_Matrix_Row (not changed). "uni_mat" is the matrix 
 of unique observations (output). "ind" is the vector of indexes of which one of the 
 unique rows the original observations correspond to (output). "N_uni" stores the 
 number of observations for each distinct covariates row (output).
***************************************************************************************/
	int ncol, nrow, k;
	
	ncol = mat.cols();
	nrow = mat.rows();

	k = 0;
	uni_mat.row(0) = mat.row(index(0));
	ind(index(0)) = k;
	N_uni(k) = 1;

	for (int i=1; i<nrow; i++) 
	{
		if (EqualArray(mat.row(index(i-1)), mat.row(index(i)), ncol)) 
		{
			ind(index(i)) = k;
			N_uni(k) += 1;
		} 
		else
		{
			k++;
			uni_mat.row(k) = mat.row(index(i));
			ind(index(i)) = k;
			N_uni(k) = 1;
		}
	}
} // Create_Uni_Matrix_Row

void indexx_Vector (const Ref<const VectorXd>& vec, Ref<VectorXi> indx) 
{	
	int n = vec.size(), i, indxt, ir=n-1, j, k, l=0, jstack=0, tmp;
	double a;
	VectorXi istack(NSTACK+1);

	for (j=0; j<n; j++) indx(j) = j;
	for (;;) 
	{
		if (ir-l < M) 
		{
			for (j=l+1; j<=ir; j++) 
			{
				indxt = indx(j);
				a = vec(indxt);
				for (i=j-1; i>=l; i--) 
				{
					if (a > vec(indx(i))) break;
					indx(i+1) = indx(i);
				}
				indx(i+1) = indxt;
			}
			if (jstack == 0) break;
			ir = istack(jstack--);
			l = istack(jstack--);
		} 
		else 
		{
			k = (l+ir) >> 1;
			tmp = indx(k);
			indx(k) = indx(l+1);
			indx(l+1) = tmp;
			if (vec(indx(l)) > vec(indx(ir))) 
			{
				tmp = indx(l);
				indx(l) = indx(ir);
				indx(ir) = tmp;
			}
			if (vec(indx(l+1)) > vec(indx(ir))) 
			{
				tmp = indx(l+1);
				indx(l+1) = indx(ir);
				indx(ir) = tmp;
			}
			if (vec(indx(l)) > vec(indx(l+1))) 
			{
				tmp = indx(l);
				indx(l) = indx(l+1);
				indx(l+1) = tmp;
			}
			i = l+1;
			j = ir;
			indxt = indx(l+1);
			a = vec(indxt);
			for (;;) 
			{
				do i++; while (vec(indx(i)) < a);
				do j--; while (vec(indx(j)) > a);
				if (j < i) break;
				tmp = indx(i);
				indx(i) = indx(j);
				indx(j) = tmp;
			}
			indx(l+1) = indx(j);
			indx(j) = indxt;
			jstack += 2;
			if (jstack > NSTACK) stdError("Error: NSTACK too small in indexx_Vector!");
			if (ir-i+1 >= j-l) 
			{
				istack(jstack) = ir;
				istack(jstack-1) = i;
				ir = j-1;
			} 
			else 
			{
				istack(jstack) = j-1;
				istack(jstack-1) = l;
				l = i;
			}
		}
	}
} // indexx_Vector

void Num_Distinct_Events (const Ref<const VectorXd>& Y, const Ref<const VectorXi>& Y_index, const Ref<const VectorXi>& Delta, int& n_event) 
{	
	double event_prev;
	
	if (Delta.sum() <= 0)
	{
		stdError("Error: No event in the dataset!");
	}
	else
	{
		if (Delta(Y_index(0)) == 1)
		{
			n_event = 1;
			event_prev = Y(Y_index(0));
		}
		else
		{
			n_event = 0;
			event_prev = -999.;
		}
		
		for(int i=0; i<Y.size()-1; i++) 
		{
			if(Y(Y_index(i)) == Y(Y_index(i+1)))
			{
				if (Delta(Y_index(i+1)) == 1 && event_prev != Y(Y_index(i+1))) 
				{
					n_event++;
					event_prev = Y(Y_index(i+1));
				}
			}
			else if (Y(Y_index(i)) < Y(Y_index(i+1)))
			{
				if (Delta(Y_index(i+1)) == 1) 
				{
					n_event++;
					event_prev = Y(Y_index(i+1));
				}
			}
			else
			{
				stdError("Error: In Num_Distinct_Events(), Y(Y_index(i)) > Y(Y_index(i+1))");
			}
		}		
	}
} // Num_Distinct_Events

void Create_Uni_Events (const Ref<const VectorXd>& Y, const Ref<const VectorXi>& Y_index, const Ref<const VectorXi>& Delta, 
	Ref<VectorXd> Y_uni_event, Ref<VectorXi> Y_risk_ind, Ref<VectorXi> Y_uni_event_n) 
{
	double event_prev;
	int k = -1, n_event = Y_uni_event.size();
	
	if (Delta(Y_index(0)) == 1)
	{
		k++;
		Y_uni_event(k) = Y(Y_index(0));
		Y_uni_event_n(k) = 1;
		event_prev = Y(Y_index(0));
	}
	else
	{
		event_prev = -999.;
	}
	
	for(int i=0; i<Y.size()-1; i++) 
	{
		if(Y(Y_index(i)) == Y(Y_index(i+1)))
		{
			if (Delta(Y_index(i+1)) == 1) 
			{
				if (event_prev != Y(Y_index(i+1)))
				{
					k++;
					Y_uni_event(k) = Y(Y_index(i+1));
					Y_uni_event_n(k) = 1;
					event_prev = Y(Y_index(i+1));		
				}
				else
				{
					Y_uni_event_n(k) += 1;
				}
			}
		}
		else if (Y(Y_index(i)) < Y(Y_index(i+1))) 
		{
			if (Delta(Y_index(i+1)) == 1) 
			{
				k++;
				Y_uni_event(k) = Y(Y_index(i+1));
				Y_uni_event_n(k) = 1;
				event_prev = Y(Y_index(i+1));
			}
		}
		else
		{
			stdError("Error: In Create_Uni_Events(), Y(Y_index(i)) > Y(Y_index(i+1))");
		}
	}
		
	if (Delta.sum() != Y_uni_event_n.sum())
	{
		stdError("Error: In Create_Uni_Events(), Delta.sum() != Y_uni_event_n.sum()");
	}
	
	if (k != n_event-1)
	{
		stdError("Error: In Create_Uni_Events(), k != n_event-1");
	}
	else
	{
		for (int i=Y.size()-1; i>-1; i--)
		{
			if (Y(Y_index(i)) >= Y_uni_event(n_event-1))
			{
				Y_risk_ind(Y_index(i)) = n_event-1;
			}
			else if (Y(Y_index(i)) < Y_uni_event(0))
			{
				Y_risk_ind(Y_index(i)) = -1;
			}
			else if (Y(Y_index(i)) >= Y_uni_event(k-1) && Y(Y_index(i)) < Y_uni_event(k))
			{
				Y_risk_ind(Y_index(i)) = k-1;
			}
			else if (Y(Y_index(i)) < Y_uni_event(k-1))
			{
				k--;
				i++;
			}
			else
			{
				stdError("Error: In Create_Uni_Events(), error in calculating Y_risk_ind()");
			}
		}
		if (k != 1)
		{
			stdError("Error: In Create_Uni_Events(), k != 1");
		}
	}
} // Create_Uni_Events

double Var (const Ref<const VectorXd>& arr) 
{
	const int n = arr.rows();
	const double mean = arr.mean();
	
	double var=0., tmp;
	
	for (int i=0; i<n; i++) 
	{
		tmp = arr(i)-mean;
		var += tmp*tmp;
	}
	
	var /= n;
	return var;
} // Var

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
		pB = Bspline_uni*logp.transpose();
		
		loglik = 0.;
		for (int i=0; i<n2; i++) {
			loglik += pB(Bspline_uni_ind(i),X_uni_ind(i));
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
