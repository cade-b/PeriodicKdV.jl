module PeriodicKdV

using LinearAlgebra, FastTransforms, SparseArrays, BandedMatrices, Plots, Colors, ApproxFun, FFTW, IterativeSolvers, LinearMaps

import Base: ==, *, +, zero, size, getindex, vcat, hcat, \, length, -, copy, setindex!

import LinearAlgebra: diag, ⋅

include("weight.jl")
include("operators.jl")
include("adaptivecauchy.jl")
include("hyperellipticsurface.jl")
include("forwardscattering.jl")
include("circlefun.jl")
include("newRHP.jl")

export cauchy, poly, transformT, transformV, transformU, transformW, SIE, SIE_new, chebV, chebU, chebW, chebT, WeightedInterval, WeightPlot, Cut, spy, FunPlot, GMRES, CauchyChop, Cauchy, M, iM, Ugrid, DefiniteIntegral, BlockVector,
aT, bT, aU, bU, aW, bW, aV, bV, HyperellipticSurface, BakerAkhiezerFunction, KdV, CircleFun, solve_rhp, rhsol



function GMRES(A,b,inner,tol,n)
    nom = a -> sqrt(abs(inner(a,a)))
    H = zeros(Complex{Float64},n+1,n)
    bnorm = nom(b)
    x = 0.
    conv_history = []
    Q = [(1.0/bnorm)*b]
    for i = 1:n
       #tic()
       #println("Operator application: ")
       #@time v = A(Q[i])
       v = A(Q[i])
       #toc()
       #tic()
       #println("Inner products: ")
       #@time
       for j = 1:i
           #tic()
           H[j,i] = inner(Q[j],v)
           #toc()
           v = v - H[j,i]*Q[j]
       end
       #println("Assembling Q:")
       H[i+1,i] = nom(v)
       Q = vcat(Q,[(1.0/H[i+1,i])*v])
       #print("Arnoldi: ")
       #toc()
       #return v
       if i > 1
           # Solve H[1:i+1,1:i]*x = bnorm*e_1, using least squares
           # TODO: Implement Givens rotations
           rhs = zeros(Float64,i+1)
           rhs[1] = bnorm
           x = H[1:i+1,1:i]\rhs
           res = norm(H[1:i+1,1:i]*x-rhs)
           conv_history = vcat(conv_history,[i,res])
           print("iteration = ")
           print(i)
           print(", residual = ")
           println(res)
           if res < tol
               return  [Q,x,conv_history]
           end
       end
    end
    println("GMRES did not terminate")
    return [Q,x,conv_history]
end

function GMRES_quiet(A,b,inner,tol,n)
    nom = a -> sqrt(abs(inner(a,a)))
    H = zeros(Complex{Float64},n+1,n)
    bnorm = nom(b)
    x = 0.
    conv_history = []
    Q = [(1.0/bnorm)*b]
    for i = 1:n
       v = A(Q[i])
       for j = 1:i
           H[j,i] = inner(Q[j],v)
           v = v - H[j,i]*Q[j]
       end
       H[i+1,i] = nom(v)
       Q = vcat(Q,[(1.0/H[i+1,i])*v])
       if i > 1
           # TODO: Implement Givens rotations?
           rhs = zeros(Float64,i+1)
           rhs[1] = bnorm
           x = H[1:i+1,1:i]\rhs
           res = norm(H[1:i+1,1:i]*x-rhs)
           conv_history = vcat(conv_history,[i,res])
           if res < tol
               return  [Q,x,conv_history]
           end
       end
    end
    println("GMRES did not terminate")
    return [Q,x,conv_history]
end

# T_n: ortho wrt to w(x) = 1/\pi 1/\sqrt(1-x^2)
#seedT = z -> 1im/(2*pi*sqrt(z^2-1))
seedT = z -> 1im/(2*pi*sqrt(z-1 |> Complex)*sqrt(z +1 |> Complex))
aT = n ->  0.0
bT = n ->  n == 0  ? 1/sqrt(2.0) : 0.5

# U_n: ortho wrt w(x) = 2/\pi \sqrt{1-x^2}
#seedU = z -> (-z + sqrt(z^2-1 |> Complex))/(pi*1im)
seedU = z -> (-z + sqrt(z-1 |> Complex)*sqrt(z +1 |> Complex))/(pi*1im)
aU = n ->  0.0
bU = n ->  0.5

# V_n: ortho wrt w(x) = 1/\pi \sqrt{1+x}/\sqrt{1-x}
seedV = z -> 1im/(2*pi)*(-1 + sqrt(-(1+z)/(1-z) |> Complex))
aV = n ->  n == 0  ? 0.5 : 0.0
bV = n ->  0.5

# W_n: ortho wrt w(x) = 1/\pi \sqrt{1-x}/\sqrt{1+x}
seedW = z ->1im/(2*pi)*(1 - sqrt(-(1-z)/(1+z) |> Complex))
aW = n ->  n == 0  ? -0.5 : 0.0
bW = n ->  0.5

## Define the weights
chebV = Weight(aV,bV,seedV,x -> 1/pi*sqrt((1+x)/(1-x)) )
chebW = Weight(aW,bW,seedW,x -> 1/pi*sqrt((1-x)/(1+x)) )
chebT = Weight(aT,bT,seedT,x -> 1/pi*1/sqrt((1+x)*(1-x)) )
chebU = Weight(aU,bU,seedU,x -> 2/pi*sqrt((1-x)*(1+x)) )

struct DefiniteIntegral <: LinearOperator
    T::Function
    a::Float64
    b::Float64
    n::Int64
end

function *(In::DefiniteIntegral,f::Function)
    In.T(map(f,M(In.a,In.b)(Ugrid(In.n))))[1]
end

function *(In::DefiniteIntegral,f::Vector)
    In.T(f)[1]
end

# Roots of Chebyshev U_n
Ugrid = n -> cos.( (2*(1:n) .- 1)/(2*n) * pi )

# Chebyshev T (orthonormalized) coefficients from values at Ugrid(n)
function transformT(x)
    out = chebyshevtransform(x)
    out[2:end] /= sqrt(2.0)
    out
end

function transformU(x)
    chebyshevutransform(x)
end

function transformV(x)
    out = chebyshevtransform(x)
    out[1] = 2*out[1]
    out[1:end-1] += out[2:end]
    out *= 0.5
    out
end

function transformW(x)
    out = chebyshevtransform(x)
    out[1] = 2*out[1]
    out[1:end-1] -= out[2:end]
    out *= 0.5
    out
end

### Polynomial evaluation matrices

function poly(a,b,n,x) # a naive use of the three-term recurrence
    p = fill(0.0im,n+1)
    p[1] = 1.0 # p_0
    p[2] = x.*p[1] - a(0)*p[1] # compute p_1
    p[2] /= b(0)
    for j = 1:n-1 # compute p_n
        p[j+2] = x.*p[j+1] - a(j)*p[j+1] - b(j-1)*p[j]
        p[j+2] /= b(j)
    end
    p
end

function poly(a,b,n,z::Vector)
    vcat(map(zz -> poly(a,b,n,zz) |> transpose , z)...)
end

### SIE setup and solve

function SIE(RHP,n)
    Cp = Cauchy(RHP,RHP,n,n,1)
    Cm = Cauchy(RHP,RHP,n,n,-1)
    m = length(RHP)
    S = fill(0.0im,m*n*2,m*n*2)
    S[1:m*n,1:m*n] = Cp
    S[m*n+1:2*m*n,m*n+1:2*m*n] = Cp

    T = copy(Cm)
    for j in 1:m
        T[(j-1)*n+1:j*n,:] *= RHP[j].J[1,2]
    end
    S[m*n+1:end,1:m*n] -= T

    T = copy(Cm)
    for j in 1:m
        T[(j-1)*n+1:j*n,:] *= RHP[j].J[1,1]
    end
    S[1:m*n,1:m*n] -= T

    T = copy(Cm)
    for j in 1:m
        T[(j-1)*n+1:j*n,:] *= RHP[j].J[2,1]
    end
    S[1:m*n,m*n+1:end] -= T

    T = copy(Cm)
    for j in 1:m
        T[(j-1)*n+1:j*n,:] *= RHP[j].J[2,2]
    end
    S[m*n+1:end,m*n+1:end] -= T

    rhs = fill(0.0im,length(RHP)*n*2)
    for j in 1:m
        rhs[(j-1)*n+1:j*n] = rhs[(j-1)*n+1:j*n] .+ (RHP[j].J[1,1] + RHP[j].J[2,1] - 1)
        rhs[m*n+(j-1)*n+1:m*n+j*n] = rhs[m*n+(j-1)*n+1:m*n+j*n] .+ (RHP[j].J[2,2] + RHP[j].J[1,2] - 1)
    end
    c = S\rhs
    f1 = [WeightFun(c[(j-1)*n+1:j*n],RHP[j].W)  for j = 1:m ]
    f2 = [WeightFun(c[n*m+(j-1)*n+1:n*m+j*n],RHP[j].W)  for j = 1:m ]
    [f1,f2,S,c]
end

function SIE_new(RHP,n,tol,max_iter=100) # no permutation
    #Cp = Cauchy(RHP,RHP,n,n,1)
    #Cm = Cauchy(RHP,RHP,n,n,-1)
    #CpBO = chop(BlockOperator(Cp,n,n),tol);
    #CmBO = chop(BlockOperator(Cm,n,n),tol);
    println("Time for generation and compression")
    @time begin
        CpBO = CauchyChop(RHP,RHP,n,n,1,tol)
        CmBO = CauchyChop(RHP,RHP,n,n,-1,tol)
    end

    println("Time for operator construction")
    @time begin
        m = length(RHP)
        Z = ZeroOperator(n)
        ZZ = BlockOperator(fill(Z,m,m))

        C⁺ = vcat(hcat(CpBO,ZZ),hcat(ZZ,CpBO))

        D11 = DiagonalBlockOperator( [-RHP[j].J[1,1] for j = 1:m], n)
        D21 = DiagonalBlockOperator( [-RHP[j].J[1,2] for j = 1:m], n)
        D12 = DiagonalBlockOperator( [-RHP[j].J[2,1] for j = 1:m], n)
        D22 = DiagonalBlockOperator( [-RHP[j].J[2,2] for j = 1:m], n)

        p = [1,m+1]
        for i = 2:m
            append!(p,i)
            append!(p,m+ i)
        end

        JC⁻ = vcat(hcat(D11*CmBO,D12*CmBO),hcat(D21*CmBO,D22*CmBO))
        S = C⁺ + JC⁻
        # notes for x derivative of solution
        # S*u = b
        # Sx*u + S*ux = bx
        # S*ux = bx - Sx*b,   Sx = JxC-

        Sp = permute(S,p)
        D = TakeDiagonalBlocks(Sp,2)

        PrS = x -> BlockVector(D\BlockVector(x,2*n),n)

        rhs = fill(0.0im,length(RHP)*n*2)
        for j in 1:m
            rhs[(j-1)*n+1:j*n] = rhs[(j-1)*n+1:j*n] .+ (RHP[j].J[1,1] + RHP[j].J[2,1] - 1)
            rhs[m*n+(j-1)*n+1:m*n+j*n] = rhs[m*n+(j-1)*n+1:m*n+j*n] .+ (RHP[j].J[2,2] + RHP[j].J[1,2] - 1)
        end

        b = BlockVector(rhs,n)
        bp = permute(b,p)

        Op = x -> PrS(Sp*x)
    end

    println("Time to solve")
    @time begin
        out = GMRES(Op,PrS(bp),⋅,tol,max_iter)
        solp = +([out[2][j]*out[1][j] for j = 1:length(out[2])]...)
        sol = ipermute(solp,p)
    end

    f1 = [WeightFun(sol[j],RHP[j].W)  for j = 1:m ]
    f2 = [WeightFun(sol[m + j],RHP[j].W)  for j = 1:m ]

    (f1,f2,S,b,Sp,bp,sol,solp,p)
end

end#module
