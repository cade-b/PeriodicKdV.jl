M = (a,b) ->  (x -> (b-a)/2*(x .+ (b+a)/(b-a)))
iM = (a,b) -> (x -> 2/(b-a)*(x .- (b+a)/2))

Ugrid = n -> cos.( (2*(1:n) .- 1)/(2*n) * pi)

#=struct ChebyParams
    a::Float64
    b::Float64
    kind::Int
end=#

J₊(z) = z-√(z-1 |> Complex)*√(z+1 |> Complex) #inverse Joukowsky map

function ChebyVIntExact(z,N::Int,a::Float64,b::Float64)
    C = zeros(ComplexF64,1,N+1)
    C[1] = (im/2π)*(-1 + sqrt((z-a)/(z-b) |> Complex))*(2/(b-a))
    for k = 1:N
        C[k+1] = C[k]*J₊(iM(a,b)(z))
    end
    C
end

function ChebyWIntExact(z,N::Int,a::Float64,b::Float64)
    C = zeros(ComplexF64,1,N+1)
    C[1] = (im/2π)*(1 - sqrt((z-b)/(z-a) |> Complex))*(2/(b-a))
    for k = 1:N
        C[k+1] = C[k]*J₊(iM(a,b)(z))
    end
    C
end

function CauchyIntervalVec(z, a::Float64, b::Float64, kind::Int, N::Int)
    if kind == 3
        C = ChebyVIntExact(z,N,a,b)
    elseif kind == 4
        C = ChebyWIntExact(z,N,a,b)
    end
C
end

#=function CauchyIntervalVec(z, X::ChebyParams, N)
    if X.kind == 1
        C = ChebyTIntExact(z,N,X.a,X.b)
    elseif X.kind == 2
        C = ChebyUIntExact(z,N,X.a,X.b)
    elseif X.kind == 3
        C = ChebyVIntExact(z,N,X.a,X.b)
    elseif X.kind == 4
        C = ChebyWIntExact(z,N,X.a,X.b)
    end
transpose(C) |> Array
end=#

function vals_to_coeffs_V(v::Vector)
    N = length(v)
    v .*= cos.(((1:N).-1/2)*pi/(2N))
    coeffs = FFTW.r2r(v,FFTW.REDFT11)/N
    coeffs
end

function vals_to_coeffs_W(v::Vector)
    N = length(v)
    v .*= sin.(((1:N).-1/2)*pi/(2N))
    coeffs = FFTW.r2r(v,FFTW.RODFT11)/N
    coeffs
end

function apply_inv(x::Vector, a::Float64, b::Float64, kind::Int)
    if kind == 3
        v = -vals_to_coeffs_W(x)*(pi/im)*(b-a)/2
    elseif kind == 4
        v = vals_to_coeffs_V(x)*(pi/im)*(b-a)/2
    end
    v
end

function apply_Cauchy_V(c::Vector,N::Int,aₒ::Float64,bₒ::Float64,aₐ::Float64,bₐ::Float64; flag=0)
    pts = M(aₐ,bₐ).(Ugrid(N)) .|> Complex
    if flag == 1
        pts .+= eps*im
    elseif flag == -1
        pts .-= eps*im
    end
    ints = @. (im/2π)*(-1 + sqrt((pts-aₒ)/(pts-bₒ) |> Complex))*(2/(bₒ-aₒ))
    vals = c[1]*ints
    jinv = J₊.(iM(aₒ,bₒ).(pts))
    for j = 1:length(c)-1
        ints .*= jinv
        if maximum(abs.(ints)) < 1e-13
            break
        end
        #vals += c[j+1]*ints
        axpy!(c[j+1],ints,vals)
    end
    vals
end

function apply_Cauchy_W(c::Vector,N::Int,aₒ::Float64,bₒ::Float64,aₐ::Float64,bₐ::Float64; flag=0)
    pts = M(aₐ,bₐ).(Ugrid(N)) .|> Complex
    if flag == 1
        pts .+= eps*im
    elseif flag == -1
        pts .-= eps*im
    end
    ints = @. (im/2π)*(1 - sqrt((pts-bₒ)/(pts-aₒ) |> Complex))*(2/(bₒ-aₒ))
    vals = c[1]*ints
    jinv = J₊.(iM(aₒ,bₒ).(pts))
    for j = 1:length(c)-1
        ints .*= jinv
        if maximum(abs.(ints)) < 1e-13
            break
        end
        #vals += c[j+1]*ints
        axpy!(c[j+1],ints,vals)
    end
    vals
end

function apply_Cauchy(c::Vector,N::Int,aₒ::Float64,bₒ::Float64,aₐ::Float64,bₐ::Float64,kind::Int; flag = 0)
    if kind == 3
        v = apply_Cauchy_V(c,N,aₒ,bₒ,aₐ,bₐ; flag=flag)
    elseif kind == 4
        v = apply_Cauchy_W(c,N,aₒ,bₒ,aₐ,bₐ; flag=flag)
    end
    v
end

w_V(x) = sqrt(x+1 |> Complex)/(pi*sqrt(1-x |> Complex))
w_W(x) = sqrt(1-x |> Complex)/(pi*sqrt(x+1 |> Complex))

function apply_inv_plus_V(x::Vector,a::Float64,b::Float64)
    weightvec = w_V.(Ugrid(length(x)))*2/(b-a)
    vals_to_coeffs_V(x./weightvec)
end

function apply_inv_plus_W(x::Vector,a::Float64,b::Float64)
    weightvec = w_W.(Ugrid(length(x)))*2/(b-a)
    vals_to_coeffs_W(x./weightvec)
end

function apply_inv_plus(x::Vector,a::Float64,b::Float64,kind::Int)
    if kind == 3
        v = apply_inv_plus_V(x,a,b)
    elseif kind == 4
        v = apply_inv_plus_W(x,a,b)
    end
    v
end

struct rhsol
    bands::Array{Float64,2}
    sol::Vector{Complex}
    nvec::Vector{Int}
    typevec::Vector{Int}
end

U(x) = [1/(x*√2) -1/(x*√2); 1/√2 1/√2]
K(α,β) = (2*U(α)'*U(β))[2,:]
J(θ,θx) = [0 θx*exp(θ); -θx*exp(-θ) 0]

function solve_many_int(bands::Array{Float64,2}, Ωs::Vector{ComplexF64}, Ωsx::Vector{ComplexF64}; nvec = Nothing, typevec = Nothing)
    g = size(bands,1)-1
    if nvec == Nothing
        nvec = 20*ones(Int,g+1)
    end
    nsum(j) = sum(nvec[1:j-1])
    
    if typevec == Nothing
        gg = (g+1)/2 |> Int
        typevec = [4*ones(Int,gg); 3*ones(Int,gg)]
    end
    
    function A_map(x) #applies preconditioned and diagonalized operator
        v = copy(x)
        @inbounds for j = 1:g+1
            vadd = zeros(ComplexF64,nvec[j])
            for k = (1:g+1)[1:end .!= j,:]
                #x1 = x[2nsum(k)+1:2nsum(k)+nvec[k]]
                #x2 = x[2nsum(k)+nvec[k]+1:2nsum(k)+2nvec[k]]

                axpy!(K(exp(Ωs[j]),exp(Ωs[k]))[1],apply_inv(apply_Cauchy(x[2nsum(k)+1:2nsum(k)+nvec[k]],nvec[j],bands[k,1],bands[k,2],bands[j,1],bands[j,2],typevec[k]),bands[j,1],bands[j,2],typevec[j]),vadd)
                axpy!(K(exp(Ωs[j]),exp(Ωs[k]))[2],apply_inv(apply_Cauchy(x[2nsum(k)+nvec[k]+1:2nsum(k)+2nvec[k]],nvec[j],bands[k,1],bands[k,2],bands[j,1],bands[j,2],typevec[k]),bands[j,1],bands[j,2],typevec[j]),vadd)
            end
            v[2nsum(j)+nvec[j]+1:2nsum(j)+2nvec[j]] += vadd
        end
        v
    end
    
    At = LinearMap(A_map, 2*sum(nvec); issymmetric=false, ismutating=false)
    #preconditoned and diagonalized RHS can be computed explicitly
    rhs = zeros(ComplexF64, 2*sum(nvec))
    @inbounds for j = 1:g+1
        if typevec[j] == 3
            rhs[2nsum(j)+nvec[j]+1] = (exp(Ωs[j])-1)*π*im*(bands[j,2]-bands[j,1])/√2
        elseif typevec[j] == 4
            rhs[2nsum(j)+nvec[j]+1] = -(exp(Ωs[j])-1)*π*im*(bands[j,2]-bands[j,1])/√2
        end
    end
    
    sol = gmres(At,rhs; reltol=1e-12)
    @inbounds for j = 1:g+1 #undo diagonalization
        x1 = sol[2nsum(j)+1:2nsum(j)+nvec[j]]
        x2 = sol[2nsum(j)+nvec[j]+1:2nsum(j)+2nvec[j]]
        sol[2nsum(j)+1:2nsum(j)+2nvec[j]] = kron(U(exp(Ωs[j]))[:,1],x1)+kron(U(exp(Ωs[j]))[:,2],x2)
    end
    ϕ(z,c) = rhsol(bands,sol,nvec,typevec)(z,c) + [1 1]
    
    # build derivative RHS
    rhsx = zeros(ComplexF64,2*nsum(g+2))
    @inbounds for j = 1:g+1
        J1, J2 = -Ωsx[j]*exp(-Ωs[j]), Ωsx[j]*exp(Ωs[j])
        x1 = zeros(ComplexF64,nvec[j])
        x2 = zeros(ComplexF64,nvec[j])
        # compute -Aₓ*sol
        for k = 1:g+1
            #x1 += J1*apply_Cauchy(sol[2nsum(k)+nvec[k]+1:2nsum(k)+2nvec[k]],nvec[j],bands[k,1],bands[k,2],bands[j,1],bands[j,2],typevec[k]; flag = -1)
            axpy!(J1,apply_Cauchy(sol[2nsum(k)+nvec[k]+1:2nsum(k)+2nvec[k]],nvec[j],bands[k,1],bands[k,2],bands[j,1],bands[j,2],typevec[k]; flag = -1), x1)
            #x2 += J2*apply_Cauchy(sol[2nsum(k)+1:2nsum(k)+nvec[k]],nvec[j],bands[k,1],bands[k,2],bands[j,1],bands[j,2],typevec[k]; flag = -1) 
            axpy!(J2,apply_Cauchy(sol[2nsum(k)+1:2nsum(k)+nvec[k]],nvec[j],bands[k,1],bands[k,2],bands[j,1],bands[j,2],typevec[k]; flag = -1), x2)
        end
        
        # add derivative of RHS
        x1 .+= J1
        x2 .+= J2
    
        # apply diagonalization
        #xx = kron(U(exp(Ωs[j]))'[:,1],x1)+kron(U(exp(Ωs[j]))'[:,2],x2)
        v1 = U(exp(Ωs[j]))'[1,1]*x1+U(exp(Ωs[j]))'[1,2]*x2
        v2 = U(exp(Ωs[j]))'[2,1]*x1+U(exp(Ωs[j]))'[2,2]*x2
        
        #apply preconditioner
        rhsx[2nsum(j)+1:2nsum(j)+2nvec[j]] = [apply_inv_plus(v1,bands[j,1],bands[j,2],typevec[j]); apply_inv(v2,bands[j,1],bands[j,2],typevec[j])]
    end
    
    solx = gmres(At,rhsx; reltol=1e-12)
    @inbounds for j = 1:g+1
        x1 = solx[2nsum(j)+1:2nsum(j)+nvec[j]]
        x2 = solx[2nsum(j)+nvec[j]+1:2nsum(j)+2nvec[j]]
        solx[2nsum(j)+1:2nsum(j)+2nvec[j]] = kron(U(exp(Ωs[j]))[:,1],x1)+kron(U(exp(Ωs[j]))[:,2],x2)
    end
    ϕx(z,c) = rhsol(bands,solx,nvec,typevec)(z,c)
    
    return ϕ,ϕx
end

function solve_rhp(x, t, BA::BakerAkhiezerFunction)
    g = length(BA.WIp)
    bands = zeros(2g,2)
    for j = 1:g
        bands[j+g,:] = [BA.WIp[j].a BA.WIp[j].b]
        bands[j,:] = [BA.WIm[j].a BA.WIm[j].b]
    end
    
    Ωs = [-im*reverse(BA.Ω(x,t)); im*BA.Ω(x,t)];
    Ωp = BA.Ω(1.0,0) - BA.Ω(0.0,0)
    Ωsx = [-im*reverse(Ωp); im*Ωp];
    
    solve_many_int(bands, Ωs, Ωsx; nvec = BA.ns);
end

function (rh::rhsol)(z,flag::Int)
    if flag == 1
        z += eps*im
    elseif flag == -1
        z -= eps*im
    else
        println("Error: Invalid parameter for flag.")
    end
    
    g = size(rh.bands,1)-1
    nsum(j) = sum(rh.nvec[1:j-1])
    
    evalvec = zeros(ComplexF64,1,sum(rh.nvec))
    coeffmat = zeros(ComplexF64,sum(rh.nvec),2)
    for j = 1:g+1
        #polys = ChebyParams(rh.bands[j,1],rh.bands[j,2],rh.typevec[j])
        evalvec[nsum(j)+1:nsum(j)+rh.nvec[j]] = CauchyIntervalVec(z,rh.bands[j,1],rh.bands[j,2],rh.typevec[j],rh.nvec[j]-1)
        coeffmat[nsum(j)+1:nsum(j)+rh.nvec[j],:] = reshape(rh.sol[2nsum(j)+1:2nsum(j)+2rh.nvec[j]],:,2)
    end
    evalvec*coeffmat
end

