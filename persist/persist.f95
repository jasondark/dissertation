program persist
    implicit none
    ! Explicit types for blas calls
    integer, parameter :: i32 = 4
    integer, parameter :: i64 = 8
    integer, parameter :: f32 = kind(1.e0)
    integer, parameter :: f64 = kind(1.d0)
    real(f64) :: ddot

    ! Iteration variables
    integer(i32) i, j
 
    ! Physical constants
    integer(i32),  parameter :: shape = 43_i32
    real(f64),     parameter :: scale = 1.91163_f64

    ! Specified constants
    real(f64)    :: kon, koff, sigma
    integer(i32) :: m, n, d

    ! Program variables
    real(f64),    allocatable :: A(:,:), x(:), y(:)
    integer(i32), allocatable :: ipiv(:)
    real(f64)                 :: z

    ! Parse command-line input
    character(len=128) :: arg
    if (command_argument_count() < 4) then
        print *, "Usage: ./persist mass minsize beta gamma"
        call exit()
    end if
    call get_command_argument(1, arg)
    read(arg,*) m
    call get_command_argument(2, arg)
    read(arg,*) n
    call get_command_argument(3, arg)
    read(arg,*) kon
    call get_command_argument(4, arg)
    read(arg,*) koff

    ! Program constants
    d = m-n+1
    sigma = 2.0_f64 * kon / koff

    ! Allocate and initialize the memory for the problem
    allocate(A(n+2, n:m), x(n:m), y(n:m), ipiv(n:m))
    ! super-diagonals
    A(2:n, n:m) = 1.0_f64
    ! diagonal
    A(n+1, n:m) = (/ (-sigma*(m-i) - (i-1), i=n, m) /)
    ! sub-diagonal
    A(n+2, n:m) = (/ (sigma*(m-i), i=n, m) /)
    ! summation vector
    x(n:(2*n-1)) = 0.0_f64
    x((2*n):m)   = (/ (i-2*n+1, i=2*n, m) /)
    ! unit vector
    y(n) = 1.0_f64
    y((n+1):m) = 0.0_f64

    ! LU-factorize the matrix operator
    call dgbtrf(d, d, 1, n-1, A, n+2, ipiv, i)

    ! update y <- inv(A)*y
    call dgbtrs('N', d, 1, n-1, 1, A, n+2, ipiv, y, d, i)

    z = ddot(d, x, 1, y, 1)

    ! update A again (maybe could update factors directly if too slow)
    ! super-diagonals
    A(2:n, n:m) = -koff*scale*1.0_f64
    ! diagonal
    A(n+1, n:m) = (/ (1 + (koff*scale)*(sigma*(m-i) + (i-1)), i=n, m) /)
    ! sub-diagonal
    A(n+2, n:m) = (/ (-(koff*scale)*sigma*(m-i), i=n, m) /)

    ! LU-factorize the matrix operator
    call dgbtrf(d, d, 1, n-1, A, n+2, ipiv, i)

    do j = 1, shape
        call dgbtrs('N', d, 1, n-1, 1, A, n+2, ipiv, y, d, i)
    end do

    print *, m, n, kon, koff, ddot(d, x, 1, y, 1)-z

    ! Cleanup the problem
    deallocate(A, x, y, ipiv)

end program persist
