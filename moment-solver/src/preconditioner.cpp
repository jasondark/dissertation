template <class uint>
class AssemblyDiagonalPreconditioner
{
	private:
		AssemblyOp<uint>* op;
    typedef double Scalar;
    typedef Matrix<Scalar,Dynamic,1> Vector;
  public:
    typedef typename Vector::StorageIndex StorageIndex;
    // this typedef is only to export the scalar type and compile-time dimensions to solve_retval
    typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;

	enum {
		ColsAtCompileTime = Dynamic,
		MaxColsAtCompileTime = Dynamic
	};

    AssemblyDiagonalPreconditioner() : m_isInitialized(false) {}

    template<typename MatType>
    explicit AssemblyDiagonalPreconditioner(const MatType& mat)     {
      compute(mat);
    }

    Index rows() const { return op->rows(); }
    Index cols() const { return op->cols(); }
    
    template<typename MatType>
    AssemblyDiagonalPreconditioner<uint>& analyzePattern(const MatType& )
    {
      return *this;
    }
    
    template<typename MatType>
    AssemblyDiagonalPreconditioner<uint>& factorize(const MatType& mat)
    {
	op = static_cast<AssemblyOp<uint>*>((void*) &mat);
	  m_isInitialized = true;
      return *this;
    }
    
    template<typename MatType>
    AssemblyDiagonalPreconditioner<uint>& compute(const MatType& mat)
    {
      return factorize(mat);
    }

    /** \internal */
    template<typename Rhs, typename Dest>
    void _solve_impl(const Rhs& b, Dest& x) const
    {
      x = op->solve_diagonal(b) ;
    }

    template<typename Rhs> inline const Solve<AssemblyDiagonalPreconditioner<uint>, Rhs>
    solve(const MatrixBase<Rhs>& b) const
    {
      eigen_assert(m_isInitialized && "DiagonalPreconditioner is not initialized.");
      eigen_assert(m_invdiag.size()==b.rows()
                && "DiagonalPreconditioner::solve(): invalid number of rows of the right hand side matrix b");
      return Solve<AssemblyDiagonalPreconditioner<uint>, Rhs>(*this, b.derived());
    }
	ComputationInfo info() { return Success; }
  protected:
    bool m_isInitialized;
};




template <class uint>
class AssemblyLowerPreconditioner
{
	private:
		AssemblyOp<uint>* op;
		double omega;

    typedef double Scalar;
    typedef Matrix<Scalar,Dynamic,1> Vector;

  public:
    typedef typename Vector::StorageIndex StorageIndex;
    // this typedef is only to export the scalar type and compile-time dimensions to solve_retval
    typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;

	enum {
		ColsAtCompileTime = Dynamic,
		MaxColsAtCompileTime = Dynamic
	};

    AssemblyLowerPreconditioner() : m_isInitialized(false) {}

    template<typename MatType>
    explicit AssemblyLowerPreconditioner(const MatType& mat)
    {
      compute(mat);
    }

    Index rows() const { return op->rows(); }
    Index cols() const { return op->cols(); }
    
    template<typename MatType>
    AssemblyLowerPreconditioner<uint>& analyzePattern(const MatType& )
    {
      return *this;
    }
    
    template<typename MatType>
    AssemblyLowerPreconditioner<uint>& factorize(const MatType& mat)
    {
		op = static_cast<AssemblyOp<uint>*>((void*) &mat);
		omega = 1.0;
	  m_isInitialized = true;
      return *this;
    }
    
    template<typename MatType>
    AssemblyLowerPreconditioner<uint>& compute(const MatType& mat)
    {
      return factorize(mat);
    }

    /** \internal */
    template<typename Rhs, typename Dest>
    void _solve_impl(const Rhs& b, Dest& x) const
    {
      x = op->solve_lower(b, omega) ;
    }

    template<typename Rhs> inline const Solve<AssemblyLowerPreconditioner<uint>, Rhs>
    solve(const MatrixBase<Rhs>& b) const
    {
      eigen_assert(m_isInitialized && "DiagonalPreconditioner is not initialized.");
      return Solve<AssemblyLowerPreconditioner<uint>, Rhs>(*this, b.derived());
    }

	ComputationInfo info() { return Success; }
	void setRelaxation(double w) { omega = w; }
  protected:
    bool m_isInitialized;
};



template <class uint>
class AssemblyUpperPreconditioner
{
	private:
		AssemblyOp<uint>* op;
		double omega;

    typedef double Scalar;
    typedef Matrix<Scalar,Dynamic,1> Vector;
	typedef AssemblyUpperPreconditioner<uint> Self;
  public:
    typedef typename Vector::StorageIndex StorageIndex;
    // this typedef is only to export the scalar type and compile-time dimensions to solve_retval
    typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;

	enum {
		ColsAtCompileTime = Dynamic,
		MaxColsAtCompileTime = Dynamic
	};

    AssemblyUpperPreconditioner() : m_isInitialized(false) {}

    template<typename MatType>
    explicit AssemblyUpperPreconditioner(const MatType& mat)
    {
      compute(mat);
    }

    Index rows() const { return op->rows(); }
    Index cols() const { return op->cols(); }
    
    template<typename MatType>
    Self& analyzePattern(const MatType& )
    {
      return *this;
    }
    
    template<typename MatType>
    Self& factorize(const MatType& mat)
    {
		op = static_cast<AssemblyOp<uint>*>((void*) &mat);
		omega = 1.0;
	  m_isInitialized = true;
      return *this;
    }
    
    template<typename MatType>
    Self& compute(const MatType& mat)
    {
      return factorize(mat);
    }


    /** \internal */
    template<typename Rhs, typename Dest>
    void _solve_impl(const Rhs& b, Dest& x) const
    {
      x = op->solve_upper(b, omega) ;
    }

    template<typename Rhs> inline const Solve<Self, Rhs>
    solve(const MatrixBase<Rhs>& b) const
    {
      eigen_assert(m_isInitialized && "DiagonalPreconditioner is not initialized.");
      return Solve<Self, Rhs>(*this, b.derived());
    }

	ComputationInfo info() { return Success; }
	void setRelaxation(double w) { omega = w; }
  protected:
    bool m_isInitialized;
};

