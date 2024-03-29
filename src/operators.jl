abstract type LinearOperator end
abstract type CompositeVector end

# Composite types and basic constructors

struct LowRankOperator{T} <: LinearOperator where T
    U::SparseMatrixCSC{T,Int64}
    S::Vector{T}
    Vt::SparseMatrixCSC{T,Int64}
end

LowRankOperator(U,S,Vt) = LowRankOperator{typeof(U[1,1])}(U,S,Vt)

effectiverank(T::LowRankOperator) = length(T.S)

struct GenericLinearOperator <: LinearOperator
   f::Function
end

struct MatrixOperator{T} <: LinearOperator where T
    A::Array{T,2}
end

effectiverank(T::MatrixOperator) = size(T.A)[1]

MatrixOperator(A) = MatrixOperator{typeof(A[1,1])}(A)

struct ZeroOperator <: LinearOperator  #TODO: Add error messages if dimensions fail to match
    n::Int64
    m::Int64
end

size(A::ZeroOperator) = (A.n,A.m)

effectiverank(T::ZeroOperator) = 0

struct ScalarOperator{T} <: LinearOperator where T
    s::T
    n::Int64
end

effectiverank(T::ScalarOperator) = 1

ScalarOperator(s,n) = ScalarOperator{typeof(s)}(s,n)

ZeroOperator(n::Integer) = ZeroOperator(n,n)

struct SumOfLinearOperators <: LinearOperator
   A::Vector{LinearOperator}
end

function Array(Op::SumOfLinearOperators)
    +(map(Array,Op.A)...)
end

struct ProductOfLinearOperators <: LinearOperator
   A::Vector{LinearOperator}
end

mutable struct BlockVector{T} <: AbstractVector{T}# supposes equal sized blocks
    ind::Vector{Int64}
    V::Vector{T}
end

BlockVector(ind,V) = BlockVector{typeof(V[1])}(ind,V)

function *(a::Number,B::LowRankOperator)  #TODO:  Add if a \approx 0 check
   LowRankOperator{typeof(a*B.U[1,1])}(B.U,a*B.S,B.Vt)
end

function *(a::Number,B::MatrixOperator)
   MatrixOperator(a*B.A)
end

function *(a::Number,B::ZeroOperator)
   B
end

function *(a::Number,B::ScalarOperator)
   ScalarOperator{typeof(a*B.s)}(a*B.s,B.n,B.m)
end

function +(A::ZeroOperator,B::LinearOperator)
   B
end

function +(A::ZeroOperator,B::ZeroOperator)
   B
end

function +(A::LinearOperator,B::ZeroOperator)
   A
end

function +(A::MatrixOperator,B::MatrixOperator)
   MatrixOperator(A.A + B.A)
end

function +(A::LinearOperator,B::LinearOperator)
    if size(A) != size(B)
        @error "Dimension Mismatch"
    end
   SumOfLinearOperators([A,B])
end

function *(A::LinearOperator,B::LinearOperator)
   ProdcutOfLinearOperators([A,B])
end

function *(L::ZeroOperator,v::Vector)
   zeros(typeof(v[1]),L.n)
end

function *(L::ScalarOperator,v::Vector)
   L.s*v
end

# TODO: see if BLAS calls speed things up
#function BLAS.gemv!(At::AbstractChar,a::Union{Bool,Number},A::ScalarOperator,x::Vector,b::Union{Bool,Number},v::Vector)
#    axpby!(a*A.s,x,b,y)
#end

function *(L::LowRankOperator,v::Vector)
   L.U*((L.S).*(L.Vt*v))
end

function *(L::SumOfLinearOperators,v::Vector)
   +(map(x -> x*v,L.A)...)
end

function *(L::ProductOfLinearOperators,v::Vector)
    w = v
    for J in reverse(L.A)
        w = J*w
    end
    w
end

function *(L::ScalarOperator,A::LinearOperator)
    L.s*A
end

function *(A::LinearOperator,L::ScalarOperator)
    L.s*A
end

function *(L::MatrixOperator,v::Vector)
   L.A*v
end

mutable struct BlockOperator
    A::Array{LinearOperator,2}
end

mutable struct DiagonalBlockOperator
    D::Vector{LinearOperator}
end

DiagonalBlockOperator(v::Vector,n::Integer) = DiagonalBlockOperator([ ScalarOperator(v[i],n) for i = 1:length(v) ])

DiagonalBlockOperator(v::Vector,n::Vector) = DiagonalBlockOperator([ ScalarOperator(v[i],n[i]) for i = 1:length(v) ])

function BlockOperator(B::Array,kk,jj)
    (n,m) = size(B)
    k = convert(Integer, n/kk |> floor)
    rk = n - kk*k
    j = convert(Integer, m/jj |> floor)
    rj = m - jj*j
    A = zeros(LinearOperator,k + (rk > 0 ? 1 : 0) ,j + (rj > 0 ? 1 : 0))
    for i = 1:k
        for l = 1:j
            A[i,l] = B[(i-1)*kk+1:i*kk,(l-1)*jj+1:l*jj] |> MatrixOperator
        end
    end
    if rj > 0
        for i = 1:k
            A[i,j+1] = B[(i-1)*kk+1:i*kk,end-rj+1:end] |> MatrixOperator
        end
    end
    if rk > 0
        for l = 1:j
            A[k+1,l] = B[end-rk+1:end,(l-1)*jj+1:l*jj] |> MatrixOperator
        end
    end
    if rk > 0 && rj > 0
        A[k+1,j+1] = B[end-rk+1:end,end-rj+1:end] |> MatrixOperator
    end
    A |> BlockOperator
end

function BlockOperator(B::DiagonalBlockOperator) #not for efficiency but for convenience
    N = length(B.D)
    Bs = zeros(LinearOperator,N,N)
    for i = 1:N
        Bs[i,i] = B.D[i]
        for j = 1:N
            if i != j
                m = size(B.D[i])[1]
                n = size(B.D[j])[2]
                Bs[i,j] = ZeroOperator(m,n)
            end
        end
    end
    Bs |> BlockOperator
end

effectiverank(B::BlockOperator) = map(effectiverank, B.A) |> sum

function getindex(B::BlockOperator,i::Integer,j::Integer)
    B.A[i,j]
end

function setindex!(B::BlockOperator,L::LinearOperator,i::Integer,j::Integer)
    B.A[i,j] = L
end

function size(A::BlockOperator)
    return A.A |> size
end

function size(A::MatrixOperator)
    return A.A |> size
end

function size(A::LowRankOperator)
   (size(A.U)[1],size(A.Vt)[2])
end

function size(A::SumOfLinearOperators)
   size(A.A[1])
end

function +(B1::BlockOperator,B2::BlockOperator)
     BlockOperator(B1.A + B2.A)
end

function *(D::DiagonalBlockOperator,B::BlockOperator)
    (n,m) = size(B)
    A = B.A |> copy
    if length(D.D) != n
        @error "Dimension mismatch"
    end
    for i = 1:n
        for j = 1:m
            A[i,j] = D.D[i]*A[i,j]
        end
    end
    BlockOperator(A)
end

function diag(B::BlockOperator)
   (n,m) = size(B)
    DiagonalBlockOperator([B[i,i] for i = 1 : min(n,m)])
end

function zero(LinearOperator)
    GenericLinearOperator( x -> 0*x)
end

function Array(A::MatrixOperator)
    A.A
end

function Array(A::ZeroOperator)
    zeros(A.n,A.m)
end

function Array(A::LowRankOperator)
    A.U*((A.S).*(A.Vt)) |> Array
end

function Array(A::BlockOperator)
    (n,m) = A.A |> size
    B = map(Array,A.A)
    hcat([ vcat(B[:,i]...) for i = 1:m]... )
end

function Array(A::DiagonalBlockOperator)
    A |> BlockOperator |> Array
    #(n,m) = A.A |> size
    #B = map(Array,A.A)
    #hcat([ vcat(B[:,i]...) for i = 1:m]... )
end

function permute_old(A::BlockOperator,p)
    (n,m) = size(A)
    if n != m
        @warn "Not square"
        return A
    end
    if length(p) != m
       @error "permutation wrong size"
    end
    B = A.A |> copy
    for i = 1:n
        for j = 1:n
            B[i,j] = A.A[p[i],p[j]]
        end
    end
    BlockOperator(B)
end

function permute(A::BlockOperator,p)
    (n,m) = size(A)
    if n != m
        @warn "Not square"
        return A
    end
    if length(p) != m
       @error "permutation wrong size"
    end
    BlockOperator(A.A[p,p])
end

function permute(V::BlockVector,p)  # Supposes equal-sized blocks
    if length(p) != length(V.ind) - 1
       @error "permutation wrong size"
    end
    BlockVector(indextogaps(V.ind)[p] |> gapstoindex,vcat(map(x -> V[x], p)...))
end

function ipermute(V::BlockVector,p)  # Supposes equal-sized blocks
    if length(p) != length(V.ind) - 1
       @error "permutation wrong size"
    end
    ip = 1:length(p) |> Array
    ip[p] = ip
    #println(ip)
    permute(V,ip)
end

function BlockVector(V::Vector,n::Integer)
    if mod(length(V),n) != 0
        @error "Unable to partition vector"
    end
    BlockVector(1:n:length(V)+1 |> Vector,V)
end

function indextogaps(v)
    v[2:end] - v[1:end-1]
end

function gapstoindex(v)
    vcat([1],1 .+ cumsum(v))
end

function BlockVector(V::BlockVector,n::Integer)
    if mod(length(V.V),n) != 0
        @error "Unable to partition vector"
    end
    BlockVector(1:n:length(V)+1 |> Vector,V.V)
end

function BlockVector!(V::BlockVector,n::Integer)
    if mod(length(V.V),n) != 0
        @error "Unable to partition vector"
    end
    V.ind = 1:n:length(V)+1 |> Vector
    V
end

function BlockVector(n::Vector{Int64},V::BlockVector)
    if length(V) != n[end]-1
        @error "Unable to partition vector"
    end
    BlockVector(n,V.V)
end

function getindex(V::BlockVector,i::Integer)
    V.V[V.ind[i]:V.ind[i+1]-1]
end

function setindex!(V::BlockVector,v::Vector,i)
    V.V[V.ind[i]:V.ind[i+1]-1] = v
end

function ⋅(W::BlockVector,V::BlockVector)
   Vector(W)'*Vector(V)
end

function *(a::Number,V::BlockVector)
    BlockVector(V.ind,a*V.V)
end

function +(W::BlockVector,V::BlockVector)
    BlockVector(W.ind,W.V + V.V)
end

function -(W::BlockVector,V::BlockVector)
    BlockVector(W.ind,W.V - V.V)
end

## TODO: Think about data storage
## I believe this function is the biggest bottleneck
function *(B::BlockOperator,V::BlockVector)  # supposes square blocks
    (m,n) = size(B)
    l = sum([ size(B[j,1])[1] for j = 1:m ])
    v = zeros(typeof(V.V[1]),l)
    ind = 0
    @inbounds for i = 1:m
        step = size(B[i,1])[1]
        ind += step
        for j = 1:n
            if typeof(B[i,j]) != ZeroOperator
                v[ind-step+1:ind] += B[i,j]*V[j]
            end
        end
    end
    BlockVector(V.ind,v)
end

#function *(B::BlockOperator,V::BlockVector)  # supposes square blocks
#    (m,n) = size(B)
#    BlockVector(V.ind,vcat([+([B.A[i,j]*V[j] for j = 1:n]...) for i = 1:m]...))
#end

function *(B::DiagonalBlockOperator,V::BlockVector) # supposes square blocks
    n = length(B.D)
    BlockVector(V.ind,vcat([B.D[j]*V[j] for j = 1:n]...))
end

function \(A::MatrixOperator,v::Vector)
    A.A\v
end

function \(B::DiagonalBlockOperator,V::BlockVector) # supposes square blocks
    n = length(B.D)
    BlockVector(V.ind,vcat([B.D[j]\V[j] for j = 1:n]...))
end

function size(V::BlockVector)
    size(V.V)
end

function length(V::BlockVector)
    length(V.V)
end

function Vector(V::BlockVector)
    V.V
end


function spy(A)
    plot(Gray.(abs.(A)))
end

spy(A::BlockOperator) = spy(A |> Array)

spy(A::DiagonalBlockOperator) = spy(A |> Array)

function vcat(A::BlockOperator,B::BlockOperator)
    BlockOperator(vcat(A.A,B.A))
end

function hcat(A::BlockOperator,B::BlockOperator)
    BlockOperator(hcat(A.A,B.A))
end

function chop(B::Array,tol)
    if tol == false
        return MatrixOperator(B)
    end
    A = svd(B)
    s = A.S
    if A.S[1] < tol
        #println("operator zeroed")
       return ZeroOperator(size(B)[1],size(B)[2])
    end
    j = length(s)
    for i = 1:length(s)
        if abs(s[i]) < tol
            j = i
            break
        end
    end
    #println(j)
    if j < length(s)/2
        LowRankOperator(A.U[:,1:j] |> sparse ,s[1:j],A.Vt[1:j,:] |> sparse)
    else
        return MatrixOperator(B)
    end
end

function copy(V::BlockVector)
    BlockVector(V.ind |> copy, V.V |> copy)
end

chop(A::MatrixOperator,tol) = chop(A.A,tol)
chop(A::GenericLinearOperator,tol) = A
chop(A::ZeroOperator,tol) = A

function chop(A::BlockOperator,tol)
    map(x -> chop(x,tol), A.A) |> BlockOperator
end

function chop(A::DiagonalBlockOperator,tol)
    map(x -> chop(x,tol), A.A) |> DiagonalBlockOperator
end

function TakeDiagonalBlock(A::BlockOperator,k::Integer,j::Integer)
    T = map(Array,A.A[k:k+j-1,k:k+j-1] |> copy)
    vcat([hcat(T[i,:]...) for i = 1:j]...) |> MatrixOperator
end

function TakeDiagonalBlocks(A::BlockOperator,j::Integer)
    (n,m) = size(A)
    if n != m
        @error "A is not square"
    end
    r = 1:j:n
    f = (x,y) -> TakeDiagonalBlock(A,x,y)
    map(f,r,fill(j,length(r))) |> DiagonalBlockOperator

end

function BlockZeroOperator(n::Vector)
    hcat( map(y -> map(x -> ZeroOperator(x,y),n),n)...) |> BlockOperator
end
