#include <Eigen/Dense>
#include "partitionlist.hpp"

using namespace Eigen;


template <class uint>
class AssemblyOp : public EigenBase< AssemblyOp<uint> > {
	private:
		const MatrixXd& A;
		const MatrixXd& B;
		const PartitionList<uint>& states;

	public:
		// eigen boilerplate
		typedef double Scalar;
		typedef double RealScalar;
		typedef int StorageIndex;
		enum {
			ColsAtCompileTime = Eigen::Dynamic,
			MaxColsAtCompileTime = Eigen::Dynamic,
			IsRowMajor = true
		};
		Index rows() const { return states.size(); }
		Index cols() const { return states.size(); }
		template <typename Rhs>
		Product< AssemblyOp<uint>, Rhs, AliasFreeProduct > operator*(const Eigen::MatrixBase<Rhs>& x) const {
			return Product< AssemblyOp<uint>, Rhs, AliasFreeProduct >(*this, x.derived()); 
		}



		AssemblyOp() {}
		AssemblyOp(const PartitionList<uint>& states_, const MatrixXd& A_, const MatrixXd& B_) : states(states_), A(A_), B(B_) {}
		//VectorXd operator*(const VectorXd& x) const { return this->apply(x); }

		VectorXd apply(const VectorXd& x) const {
			VectorXd b = VectorXd::Zero(x.size());

			uint index = 0, nbrIndex, n, m, len = (*(states.cbegin())).getLength();
			Partition<uint> nbr;

			for (const auto& state : states) {
				for (size_t i = 1; i <= len; i++) { // size to react with
					n = state.get(i);
					if (n == 0) // make sure we can react
						continue;

					for (size_t j = 1; j <= i-1; j++) {
						// fragment
						nbr = state;
						nbr.decrement(i);
						nbr.increment(j);
						nbr.increment(i-j);
						nbrIndex = states.indexOf(nbr);
						b(index) += B(j-1,i-j-1) * (double) n * (x(index) - x(nbrIndex));

						// aggregate
						m = state.get(j);
						if (m == 0)
							continue;

						if (i+j <= len) {
							nbr = state;
							nbr.decrement(i);
							nbr.decrement(j);
							nbr.increment(i+j);
							nbrIndex = states.indexOf(nbr);
							b(index) += A(i-1,j-1) * (double) (n * m) * (x(index) - x(nbrIndex));
						}
						else {
							b(index) += A(i-1,j-1) * (double) (n * m) * x(index);
						}
					}

					if (n > 1) {
						if (2*i <= len) {
							nbr = state;
							nbr.decrement(i);
							nbr.decrement(i);
							nbr.increment(2*i);
							nbrIndex = states.indexOf(nbr);
							b(index) += A(i-1,i-1) * (double) (n*(n-1)) * (x(index) - x(nbrIndex));
						}
						else {
							b(index) += A(i-1,i-1) * (double) (n*(n-1)) * x(index);
						}
					}

					for (size_t j = i+1; j <= len; j++) {
						// aggregate
						m = state.get(j);
						if (m == 0)
							continue;

						if (i+j <= len) {
							nbr = state;
							nbr.decrement(i);
							nbr.decrement(j);
							nbr.increment(i+j);
							nbrIndex = states.indexOf(nbr);
							b(index) += A(i-1,j-1) * (double) (n * m) * (x(index) - x(nbrIndex));
						}
						else {
							b(index) += A(i-1,j-1) * (double) (n * m) * x(index);
						}
					}
				}

				++index;
			}
			return b;
		}

		VectorXd adjointApply(const VectorXd& x) const {
			VectorXd b = VectorXd::Zero(x.size());

			uint index = 0, nbrIndex, n, m, len = (*(states.cbegin())).getLength();
			Partition<uint> nbr;
			double rate;

			for (const auto& state : states) {
				for (size_t i = 1; i <= len; i++) { // size to react with
					n = state.get(i);
					if (n == 0) // make sure we can react
						continue;

					for (size_t j = 1; j <= i-1; j++) {
						// fragment
						nbr = state;
						nbr.decrement(i);
						nbr.increment(j);
						nbr.increment(i-j);
						nbrIndex = states.indexOf(nbr);
						rate = B(j-1,i-j-1) * (double) n * x(index);
						b(index) += rate;
						b(nbrIndex) -= rate;

						// aggregate
						m = state.get(j);
						if (m == 0)
							continue;

						if (i+j <= len) {
							nbr = state;
							nbr.decrement(i);
							nbr.decrement(j);
							nbr.increment(i+j);
							nbrIndex = states.indexOf(nbr);
							rate = A(i-1,j-1) * (double) (n * m) * x(index);
							b(index) += rate;
							b(nbrIndex) -= rate;
						}
						else {
							rate = A(i-1,j-1) * (double) (n * m) * x(index);
							b(index) += rate;
						}
					}

					if (n > 1) {
						if (2*i <= len) {
							nbr = state;
							nbr.decrement(i);
							nbr.decrement(i);
							nbr.increment(2*i);
							nbrIndex = states.indexOf(nbr);
							rate = A(i-1,i-1) * (double) (n*(n-1)) * x(index);
							b(index) += rate;
							b(nbrIndex) -= rate;
						}
						else {
							rate = A(i-1,i-1) * (double) (n*(n-1)) * x(index);
							b(index) += rate;
						}
					}

					for (size_t j = i+1; j <= len; j++) {
						// aggregate
						m = state.get(j);
						if (m == 0)
							continue;

						if (i+j <= len) {
							nbr = state;
							nbr.decrement(i);
							nbr.decrement(j);
							nbr.increment(i+j);
							nbrIndex = states.indexOf(nbr);
							rate = A(i-1,j-1) * (double) (n * m) * x(index);
							b(index) += rate;
							b(nbrIndex) -= rate;
						}
						else {
							rate = A(i-1,j-1) * (double) (n * m) * x(index);
							b(index) += rate;
						}
					}
				}

				++index;
			}
			return b;
		}
/*
		Matrix<double,Dynamic,4> split_apply(const VectorXd& x) const {
			Matrix<double,Dynamic,4> b(x.size(),4);
			b.setConstant(0.0);
			
			uint index = 0, nbrIndex, n, m, len = (*(states.cbegin())).getLength();
			Partition<uint> nbr;

			double rate = 0.0;
			for (const auto& state : states) {
				for (size_t i = 1; i <= len; i++) { // size to react with
					n = state.get(i);
					if (n == 0) // make sure we can react
						continue;

					for (size_t j = 1; j <= i-1; j++) {
						// fragment
						nbr = state;
						nbr.decrement(i);
						nbr.increment(j);
						nbr.increment(i-j);
						nbrIndex = states.indexOf(nbr);

						rate = B(j-1,i-j-1) * (double) n;
						b(index,0) -= rate * x(nbrIndex);
						b(index,1) += rate * x(index);

						// aggregate
						m = state.get(j);
						if (m == 0)
							continue;

						rate = A(i-1,j-1) * (double) (n * m);
						b(index,2) += rate * x(index);
						if (i+j <= len) {
							nbr = state;
							nbr.decrement(i);
							nbr.decrement(j);
							nbr.increment(i+j);
							nbrIndex = states.indexOf(nbr);
							b(index,3) -= rate * x(nbrIndex);
						}
					}

					if (n > 1) {
						rate = A(i-1,i-1) * (double) (n*(n-1));
						b(index,2) += rate * x(index);
						if (2*i <= len) {
							nbr = state;
							nbr.decrement(i);
							nbr.decrement(i);
							nbr.increment(2*i);
							nbrIndex = states.indexOf(nbr);
							b(index,3) -= rate * x(nbrIndex);
						}
					}

					for (size_t j = i+1; j <= len; j++) {
						// aggregate
						m = state.get(j);
						if (m == 0)
							continue;

						rate = A(i-1,j-1) * (double) (n * m);
						b(index,2) += rate * x(index);
						if (i+j <= len) {
							nbr = state;
							nbr.decrement(i);
							nbr.decrement(j);
							nbr.increment(i+j);
							nbrIndex = states.indexOf(nbr);
							b(index,3) -= rate * x(nbrIndex);
						}
					}
				}

				++index;
			}
			return b;
		}*/

		VectorXd solve_diagonal(const VectorXd& b) const {
			VectorXd x = b;

			uint index = 0, nbrIndex, n, m, len = (*(states.cbegin())).getLength();
			Partition<uint> nbr;
			double currentRate, totalRate = 0.0;

			for (const auto& state : states) {
				for (size_t i = 1; i <= len; i++) { // size to react with
					n = state.get(i);
					if (n == 0) // make sure we can react
						continue;

					for (size_t j = 1; j <= i-1; j++) {
						// fragment
						totalRate += B(j-1,i-j-1) * (double) n;

						// aggregate
						m = state.get(j);
						totalRate += A(i-1,j-1) * (double) (n * m);
					}

					totalRate += A(i-1,i-1) * (double) (n*(n-1));

					for (size_t j = i+1; j <= len; j++) {
						// aggregate
						m = state.get(j);
						totalRate += A(i-1,j-1) * (double) (n * m);
					}
				}
				x(index) /= totalRate;
				++index;
			}
			return x;
		}



		VectorXd solve_lower(const VectorXd& b, double omega) const {
			VectorXd x = b;

			uint index = 0, nbrIndex, n, m, len = (*(states.cbegin())).getLength();
			Partition<uint> nbr;
			double currentRate, totalRate;

			for (const auto& state : states) {
				totalRate = 0.0;
				for (size_t i = 1; i <= len; i++) { // size to react with
					n = state.get(i);
					if (n == 0) // make sure we can react
						continue;

					for (size_t j = 1; j <= i-1; j++) {
						// fragment
						nbr = state;
						nbr.decrement(i);
						nbr.increment(j);
						nbr.increment(i-j);
						nbrIndex = states.indexOf(nbr);
						currentRate = B(j-1,i-j-1) * (double) n;
						totalRate += currentRate;
						x(index) += currentRate * x(nbrIndex);

						// aggregate
						m = state.get(j);
						totalRate += A(i-1,j-1) * (double) (n * m);
					}

					totalRate += A(i-1,i-1) * (double) (n*(n-1));

					for (size_t j = i+1; j <= len; j++) {
						// aggregate
						m = state.get(j);
						totalRate += A(i-1,j-1) * (double) (n * m);
					}
				}
				x(index) /= totalRate/omega;
				++index;
			}
			return x;
		}

		VectorXd solve_upper(const VectorXd& b, double omega) const {
			VectorXd x = b;

			uint index = 0, nbrIndex, n, m, len = (*(states.cbegin())).getLength();
			Partition<uint> nbr;
			double currentRate, totalRate;

			index = states.size();
			auto stateit = states.end();
			while (index-->0) {
				--stateit;

				auto state = *stateit;
				totalRate = 0.0;
				for (size_t i = 1; i <= len; i++) { // size to react with
					n = state.get(i);
					if (n == 0) // make sure we can react
						continue;

					for (size_t j = 1; j <= i-1; j++) {
						// fragment
						totalRate += B(j-1,i-j-1) * (double) n;

						// aggregate
						m = state.get(j);
						currentRate = A(i-1,j-1) * (double) (n * m);
						totalRate += currentRate;

						if (i+j <= len) {
							nbr = state;
							nbr.decrement(i);
							nbr.decrement(j);
							nbr.increment(i+j);
							nbrIndex = states.indexOf(nbr);
							x(index) += currentRate * x(nbrIndex);
						}
					}

					if (n > 1) {
						currentRate = A(i-1,i-1) * (double) (n*(n-1));
						totalRate += currentRate;
						if (2*i <= len) {
							nbr = state;
							nbr.decrement(i);
							nbr.decrement(i);
							nbr.increment(2*i);
							nbrIndex = states.indexOf(nbr);
							x(index) += currentRate * x(nbrIndex);
						}
					}

					for (size_t j = i+1; j <= len; j++) {
						// aggregate
						m = state.get(j);
						if (m == 0)
							continue;

						currentRate = A(i-1,j-1) * (double) (n * m);
						totalRate += currentRate;
						if (i+j <= len) {
							nbr = state;
							nbr.decrement(i);
							nbr.decrement(j);
							nbr.increment(i+j);
							nbrIndex = states.indexOf(nbr);
							x(index) += currentRate * x(nbrIndex);
						}
					}
				}
				x(index) /= totalRate/omega;
			}
			return x;
		}
};



namespace Eigen {
namespace internal {
  template<typename uint>
  struct traits< AssemblyOp<uint> > :  public Eigen::internal::traits<Eigen::SparseMatrix<double> > {};
}
}





// Implementation of MatrixReplacement * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen {
namespace internal {
  template<typename Rhs, typename uint>
  struct generic_product_impl<AssemblyOp<uint>, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
  : generic_product_impl_base<AssemblyOp<uint>,Rhs,generic_product_impl<AssemblyOp<uint>,Rhs> >
  {
    typedef typename Product<AssemblyOp<uint>,Rhs>::Scalar Scalar;
    template<typename Dest>
    static void scaleAndAddTo(Dest& dst, const AssemblyOp<uint>& lhs, const Rhs& rhs, const Scalar& alpha)
    {
      // This method should implement "dst += alpha * lhs * rhs" inplace,
      // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
      assert(alpha==Scalar(1) && "scaling is not implemented");
      // Here we could simply call dst.noalias() += lhs.my_matrix() * rhs,
      // but let's do something fancier (and less efficient):
      dst.noalias() += lhs.apply(rhs.matrix());
    }
  };
}
}
