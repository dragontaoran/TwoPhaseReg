#ifndef UTILITY_H
#define UTILITY_H

#include <string>
#include <RcppEigen.h>

using namespace std;
using namespace Eigen;

void stdError (const string reason);
bool BigArray (const Ref<const RowVectorXd>& arr1, const Ref<const RowVectorXd>& arr2, int p);
bool SmallArray (const Ref<const RowVectorXd>& arr1, const Ref<const RowVectorXd>& arr2, int p);
bool EqualArray (const Ref<const RowVectorXd>& arr1, const Ref<const RowVectorXd>& arr2, int p);
void indexx_Matrix_Row (const Ref<const MatrixXd>& mat, Ref<VectorXi> indx);
void Num_Uni_Matrix_Row (const Ref<const MatrixXd>& mat, const Ref<const VectorXi>& index, int& n1);
void Create_Uni_Matrix_Row (const Ref<const MatrixXd>& mat, const Ref<const VectorXi>& index, Ref<MatrixXd> uni_mat,
	Ref<VectorXi> ind, Ref<VectorXi> N_uni);
void indexx_Vector (const Ref<const VectorXd>& vec, Ref<VectorXi> indx);

/***************************************************************************************
 Find the number of unique events, and return it as "n_event". 
 "Y" is the vector of times. 
 "Y_index" is the index vector of Y output by "indexx_Vector()". 
 "Delta" is the vector of event indicators.
***************************************************************************************/
void Num_Distinct_Events (const Ref<const VectorXd>& Y, const Ref<const VectorXi>& Y_index, const Ref<const VectorXi>& Delta, int& n_event);

/***************************************************************************************
 Create unique events. 
 "Y" is the vector of times. 
 "Y_index" is the index vector of Y output by "indexx_Vector()". 
 "Delta" is the vector of event indicators. 
 "Y_uni_event" is a vector of unique events.
 "Y_risk_ind" is a vector of indexes to which one of the risk sets each element in "Y" corresponds.
 "Y_uni_event_n" is a vector of numbers of events corresponding to each unique event time.
 "L" is the vector of left-truncation times.
 "L_index" is the index vector of L output by "indexx_Vector()".
 "L_risk_ind" is a vector of indexes to which one of the risk sets each element in "L" corresponds.
***************************************************************************************/
void Create_Uni_Events (const Ref<const VectorXd>& Y, const Ref<const VectorXi>& Y_index, const Ref<const VectorXi>& Delta, 
	Ref<VectorXd> Y_uni_event, Ref<VectorXi> Y_risk_ind, Ref<VectorXi> Y_uni_event_n);
void Create_Uni_Events_LeftTrunc (const Ref<const VectorXd>& Y, const Ref<const VectorXd>& L, const Ref<const VectorXi>& Y_index, const Ref<const VectorXi>& L_index,
	const Ref<const VectorXi>& Delta, Ref<VectorXd> Y_uni_event, Ref<VectorXi> Y_risk_ind, Ref<VectorXi> Y_uni_event_n, Ref<VectorXi> L_risk_ind);

double Var(const Ref<const VectorXd>& arr);
double WaldLinearGeneralSplineProfile (Ref<MatrixXd> pB, Ref<RowVectorXd> p_col_sum,
	Ref<VectorXd> q_row_sum, Ref<MatrixXd> p, Ref<MatrixXd> p0, Ref<MatrixXd> P_theta, Ref<MatrixXd> q, Ref<VectorXd> resi_n, Ref<MatrixXd> logp,
	const Ref<const VectorXd>& theta, const Ref<const VectorXd>& Y, const Ref<const MatrixXd>& X, const Ref<const MatrixXd>& Bspline_uni,
	const Ref<const MatrixXd>& ZW, const Ref<const MatrixXd>& X_uni, const Ref<const VectorXi>& X_uni_ind, 
	const Ref<const VectorXi>& Bspline_uni_ind, const Ref<const MatrixXd>& p_static, const double sigma_sq, const int n, const int n2, const int m, 
	const int s, const int n_minus_n2, const int X_nc, const int ZW_nc, const int MAX_ITER, const double TOL);
void WaldLinearVarianceMLE0 (Ref<MatrixXd> cov_theta, const Ref<const MatrixXd>& LS_XtX, const Ref<const VectorXd>& LS_XtY, const Ref<const VectorXd>& p,
	const Ref<const VectorXd>& Y, const Ref<const MatrixXd>& W, const Ref<const VectorXd>& theta, const Ref<const MatrixXd>& X_uni, const Ref<const MatrixXd>& q, 
	const int n_minus_n2, const int X_nc, const int W_nc, const int m, const int ncov, const int n, const int n2, const double sigma_sq);
double WaldCoxphGeneralSplineProfile (Ref<MatrixXd> pB, Ref<RowVectorXd> p_col_sum, Ref<VectorXd> ZW_theta, Ref<VectorXd> X_uni_theta, Ref<MatrixXd> e_X_uni_theta,
	Ref<VectorXd> e_X_theta, Ref<VectorXd> lambda, Ref<VectorXd> lambda0, Ref<VectorXd> Lambda,
	Ref<VectorXd> q_row_sum, Ref<MatrixXd> p, Ref<MatrixXd> p0, Ref<MatrixXd> P_theta, Ref<MatrixXd> q, Ref<MatrixXd> logp,
	const Ref<const VectorXd>& theta, const Ref<const VectorXd>& Y, const Ref<const VectorXi>& Delta, const Ref<const MatrixXd>& X, 
	const Ref<const MatrixXd>& Bspline_uni, const Ref<const MatrixXd>& ZW, const Ref<const MatrixXd>& X_uni, const Ref<const VectorXi>& X_uni_ind, 
	const Ref<const VectorXi>& Bspline_uni_ind, const Ref<const VectorXd>& Y_uni_event, const Ref<const VectorXi>& Y_uni_event_n, 
	const Ref<const VectorXi>& Y_risk_ind, const Ref<const MatrixXd>& p_static, const int n, const int n2, const int m, const int n_event_uni, 
	const int s, const int n_minus_n2, const int X_nc, const int ZW_nc, const int MAX_ITER, const double TOL);
double WaldCoxphGeneralSplineProfileLeftTrunc (Ref<MatrixXd> pB, Ref<RowVectorXd> p_col_sum, Ref<VectorXd> ZW_theta, Ref<VectorXd> X_uni_theta, Ref<MatrixXd> e_X_uni_theta,
	Ref<VectorXd> e_X_theta, Ref<VectorXd> lambda, Ref<VectorXd> lambda0, Ref<VectorXd> Lambda,
	Ref<VectorXd> q_row_sum, Ref<MatrixXd> p, Ref<MatrixXd> p0, Ref<MatrixXd> P_theta, Ref<MatrixXd> q, Ref<MatrixXd> logp,
	const Ref<const VectorXd>& theta, const Ref<const VectorXd>& Y, const Ref<const VectorXd>& L, const Ref<const VectorXi>& Delta, const Ref<const MatrixXd>& X, 
	const Ref<const MatrixXd>& Bspline_uni, const Ref<const MatrixXd>& ZW, const Ref<const MatrixXd>& X_uni, const Ref<const VectorXi>& X_uni_ind, 
	const Ref<const VectorXi>& Bspline_uni_ind, const Ref<const VectorXd>& Y_uni_event, const Ref<const VectorXi>& Y_uni_event_n, 
	const Ref<const VectorXi>& Y_risk_ind, const Ref<const VectorXi>& L_risk_ind, const Ref<const MatrixXd>& p_static, const int n, const int n2, const int m, const int n_event_uni, 
	const int s, const int n_minus_n2, const int X_nc, const int ZW_nc, const int MAX_ITER, const double TOL);	
#endif
