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
	const Ref<const MatrixXd>& Z, const Ref<const MatrixXd>& W, const Ref<const MatrixXd>& X_uni, const Ref<const VectorXi>& X_uni_ind, 
	const Ref<const VectorXi>& Bspline_uni_ind, const Ref<const MatrixXd>& p_static, const double sigma_sq, const int n, const int n2, const int m, 
	const int s, const int n_minus_n2, const int X_nc, const int Z_nc, const int W_nc, const int MAX_ITER, const double TOL) 
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
	P_theta.col(0).noalias() -= Z.bottomRows(n_minus_n2)*theta.segment(X_nc,Z_nc);
	P_theta.col(0).noalias() -= W.bottomRows(n_minus_n2)*theta.tail(W_nc);
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
	p.setConstant(1./m);
	p0.setConstant(1./m);
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
		resi_n.noalias() -= Z.topRows(n2)*theta.segment(X_nc,Z_nc);
		resi_n.noalias() -= W.topRows(n2)*theta.tail(W_nc);
		
		tmp = resi_n.squaredNorm();
		tmp /= 2.*sigma_sq;
		loglik -= tmp;
		/**** calculate the likelihood *************************************************************************************************************/
		
		return loglik;	
	}
} // WaldLinearGeneralSplineProfile

void WaldLinearVarianceMLE0 (Ref<MatrixXd> cov_theta, const Ref<const MatrixXd>& LS_XtX, const Ref<const VectorXd>& LS_XtY, const Ref<const VectorXd>& p,
	const Ref<const VectorXd>& Y, const Ref<const MatrixXd>& W, const Ref<const VectorXd>& theta, const Ref<const MatrixXd>& X_uni, const Ref<const MatrixXd>& q, 
	const int n_minus_n2, const int X_nc, const int W_nc, const int m, const int ncov, const int n, const int n2, const double sigma_sq) 
{
	const int dimQ = ncov+1+m;

	VectorXd l1i(dimQ);
	VectorXd l1ik(dimQ);
	MatrixXd Q(dimQ, dimQ);
	MatrixXd resi(n_minus_n2, m);
	MatrixXd inv_profile_mat(dimQ-1, dimQ-1);
	
	/**** calculate resi ***************************************************************************************************************************/
	resi.col(0) = Y.tail(n_minus_n2);
	resi.col(0).noalias() -= W.bottomRows(n_minus_n2)*theta.tail(W_nc);

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
			l1ik.segment(X_nc,W_nc) = resi(i,k)*(W.row(i+n2).transpose())/sigma_sq;
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
}
