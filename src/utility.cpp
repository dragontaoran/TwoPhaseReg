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

void Create_Uni_Events_LeftTrunc (const Ref<const VectorXd>& Y, const Ref<const VectorXd>& L, const Ref<const VectorXi>& Y_index, const Ref<const VectorXi>& L_index,
	const Ref<const VectorXi>& Delta, Ref<VectorXd> Y_uni_event, Ref<VectorXi> Y_risk_ind, Ref<VectorXi> Y_uni_event_n, Ref<VectorXi> L_risk_ind) 
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
			stdError("Error: In Create_Uni_Events_LeftTrunc(), Y(Y_index(i)) > Y(Y_index(i+1))");
		}
	}
		
	if (Delta.sum() != Y_uni_event_n.sum())
	{
		stdError("Error: In Create_Uni_Events_LeftTrunc(), Delta.sum() != Y_uni_event_n.sum()");
	}
	
	if (k != n_event-1)
	{
		stdError("Error: In Create_Uni_Events_LeftTrunc(), k != n_event-1");
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
				stdError("Error: In Create_Uni_Events_LeftTrunc(), error in calculating Y_risk_ind()");
			}
		}
		if (k != 1)
		{
			stdError("Error: In Create_Uni_Events_LeftTrunc(), k != 1 when calculating Y_risk_ind");
		}
		
		k = n_event-1;
		for (int i=L.size()-1; i>-1; i--)
		{
			if (L(L_index(i)) >= Y_uni_event(n_event-1))
			{
				L_risk_ind(L_index(i)) = n_event-1;
			}
			else if (L(L_index(i)) < Y_uni_event(0))
			{
				L_risk_ind(L_index(i)) = -1;
			}
			else if (L(L_index(i)) >= Y_uni_event(k-1) && L(L_index(i)) < Y_uni_event(k))
			{
				L_risk_ind(L_index(i)) = k-1;
			}
			else if (L(L_index(i)) < Y_uni_event(k-1))
			{
				k--;
				i++;
			}
			else
			{
				stdError("Error: In Create_Uni_Events_LeftTrunc(), error in calculating L_risk_ind()");
			}
		}
	}
} // Create_Uni_Events_LeftTrunc

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
