mutable struct HyperellipticSurface
    #A::Array{Complex{Float64},2}  # Matrix of a-periods of basis differentials
    gaps::Array{Float64,2}  # gaps between cuts, suppose that these number are all positive, one band starts at zero
    g::Int64 # genus
    D::Array{Float64,2} # a simple divisor of points in the gaps, second entry \pm 1 for sheet
    Ω0::Vector{Complex{Float64}} # Abel map of D - D' where D' = gaps[:,1]
    Ωx::Vector{Complex{Float64}}
    Ωt::Vector{Complex{Float64}}
    E # a special constant
    α1
end

function HyperellipticSurface(q0::Function,L::Float64,n=100,tol=1e-14,m=200,trunctol=1e-14,invtol=.4)
    gaps, zs, α1 = ScatteringData(q0,L,n,tol,trunctol)
    k = size(gaps)[1]
    S = HyperellipticSurface(gaps,zs,α1,m)
    RefineHyperellipticSurface!(q0,L,S,invtol,n,tol,k,m)
    return S
end

function HyperellipticSurface(gaps,zs,α1,m=50;cycleflag = false)
    ep = 0 #TODO: Probably a good idea to eliminate the need for this.
    r = (x,y) -> (x |> Complex |> sqrt)/(y |> Complex |> sqrt)
    pr = (x,y) -> (x |> Complex |> sqrt)*(y |> Complex |> sqrt)
    #p = (z) -> -1im*prod(map(r,gaps[:,1] .- z, gaps[:,2] .- z))/sqrt(-z |> Complex)
    #P = (z) -> prod(map(pr, z .- gaps[:,1],  z .- gaps[:,2]))*1im*sqrt(-z |> Complex)
    bands = copy(gaps)
    bands[:,2] = gaps[:,1]
    bands[2:end,1] = gaps[1:end-1,2]
    bands[1,1] = 0.0

    # TODO: The biggest bottle neck in this whole thing is probably the evaluation of F.
    # Right now it is evaluated independently.  Should be able to iteration and use
    # F(j,k,z) to compute F(j',k',z) for neighboring j',k'
    function F(j,k,z)
        out = -1im/sqrt(-z |> Complex)
        for i = 1:size(gaps)[1]
            if i != j && i != k
                out *= r(z - gaps[i,1], z - gaps[i,2])
            end
        end
        if j == k
            return -out*pi*1im
        else
            return -1im*(gaps[k,2]-gaps[k,1])/2*out/pr(z - gaps[j,1], z - gaps[j,2])*pi
        end
    end

	function F(j,k,z,l)
        out = -1im/sqrt(-z |> Complex)
		out *= sqrt(z)^l
        for i = 1:size(gaps)[1]
            if i != j && i != k
                out *= r(z - gaps[i,1], z - gaps[i,2])
            end
        end
        if j == k
            return -out*pi*1im
        else
            return -1im*(gaps[k,2]-gaps[k,1])/2*out/pr(z - gaps[j,1], z - gaps[j,2])*pi
        end
    end


#     function G(j,k,z)
#         out = -1im/sqrt(gaps[end,end]-z |> Complex)
#         for i = 1:size(bands)[1]
#             if i != j && i != k
#                 out *= r(z - bands[i,2], z - bands[i,1])
#             end
#         end
#         if j == k
#             return -out*pi*1im
#         else
#             return 1im*(bands[k,2]-bands[k,1])/2*out/pr(z - bands[j,1], z - bands[j,2])*pi
#         end
#     end

    function γ(z)
        out = -1im/sqrt(-z |> Complex)
        for i = 1:size(gaps)[1]
            out *= r(z - gaps[i,1], z - gaps[i,2])
        end
        if imag(z) >= 0.0
            return out
        else
            return -out
        end
    end

	function γ(j,z)
        out = -1im/sqrt(-z |> Complex)
        for i = 1:size(gaps)[1]
			if i != j
            	out *= r(z - gaps[i,1], z - gaps[i,2])
			end
        end
        if imag(z) >= 0.0
            return out
        else
            return -out
        end
    end
	g = size(gaps)[1]
	gen_abel_pts = (a,b,λ) -> M(a,b)(Ugrid(m))
	abel_pts = map(gen_abel_pts,gaps[:,1],gaps[:,2],zs[:,1])
	abel_γ =  copy(abel_pts)
	for i = 1:g
		abel_γ[i] = map(z -> γ(i,z), abel_pts[i])
	end

    A = zeros(Complex{Float64},g,g);

	function J(n,λ)
        if λ > 1 && n == 0
            return 1.0
        elseif λ > 1
            return 0
        end

        if λ < -1
            return 0.0
        end

        if n == 0
            1 - acos(λ)/pi
        else
            - sqrt(2)/pi*sin(n*acos(λ))/n
        end
    end

    function J(f,a,b,n,λ)
        cs = transformT(map(f,M(a,b)(Ugrid(n))))
		#display(cs[end] |> abs)
	    out = 0im
		if abs(a - λ) < 1e-14
			return cs[1]
		elseif abs(b - λ) < 1e-14
			return out
		else
	    	for i = 1:length(cs)
	        	out += cs[i]*J(i-1,iM(a,b)(λ))
	    	end
		end
	    cs[1] - out
    end

	function J(v,a,b,λ)
	    cs = transformT(v)
	    #display(cs[end] |> abs)
	    out = 0im
		if abs(a - λ) < 1e-14
			return cs[1]
		elseif abs(b - λ) < 1e-14
			return out
		else
	    	for i = 1:length(cs)
	        	out += cs[i]*J(i-1,iM(a,b)(λ))
	    	end
		end
	    cs[1] - out
	end


	#if new[2]


	function Abelvec(n,j,k,λ) # integrate differential k over part of gap j
		a = gaps[j,1]
		b = gaps[j,2]
		if k == j
			-J(xx -> F(j,k,xx), a, b, n, λ)
		else
			vals = copy(abel_γ[j])
			#vals .*= map(r,abel_pts[j] .- b, abel_pts[j] .- a)
			vals .*= (abel_pts[j] .- a)./(abel_pts[j] .- gaps[k,1])
			1im*pi*J(vals,a,b,λ)
		end
	end

	function Abelvec(n,k,λ) # integrate differential k over part of gap j
		j = 1
		for jj = 1:g
			if gaps[j,1] <= λ <= gaps[j,2]
				break
			end
			j += 1
		end
		if j == g + 1
			#@warn "Not in a gap"
			return 0
		else
			return Abelvec(n,j,k,λ)
		end
	end

	function powervec(n,j,k,λ,l) # integrate differential k over part of gap j
		a = gaps[j,1]
		b = gaps[j,2]
		if k == j
			-J(xx -> F(j,k,xx,l), a, b, n, λ)
		else
			vals = copy(abel_γ[j]).*(sqrt.(abel_pts[j]).^l)
			#vals .*= map(r,abel_pts[j] .- b, abel_pts[j] .- a)
			vals .*= (abel_pts[j] .- a)./(abel_pts[j] .- gaps[k,1])
			1im*pi*J(vals,a,b,λ)
		end
	end

	function powervec(n,k,λ,l) # integrate differential k over part of gap j
		j = 1
		for jj = 1:g
			if gaps[j,1] <= λ <= gaps[j,2]
				break
			end
			j += 1
		end
		if j == g + 1
			#@warn "Not in a gap"
			return 0
		else
			return powervec(n,j,k,λ,l)
		end
	end

	if cycleflag
    	as = (gaps[:,1] + gaps[:,2])/2 |> complex
		if size(gaps)[1] > 1
        	dist = (gaps[2:end,2] - gaps[1:end-1,1])/2 |> minimum # no need to be uniform here
	    	dist = min(dist, gaps[1,1]/2)
		else
			dist = gaps[1,1]/2
		end
    	rads = (gaps[:,2] - gaps[:,1])/2 .+ dist
    	cfuns = (a,rad) -> CircleFun(z -> map(γ,z), a, rad, m)
    	γs = map(cfuns,as,rads)

    	for i = 1:g
        	b = gaps[i,1]
        	divfun = ff -> CircleFun(ff.a,ff.r,ff.v./(circle_points(ff) .- b))
        	df = map(divfun,γs)
        	A[i,:] = map(Integrate,df)
    	end
		#display(A)
    #end

#     tB = zeros(Complex{Float64},g,g);
#     for i = 1:g
#         for j = 1:g
#             if i == j
#                 tB[i,i] = 2*(DefiniteIntegral(transformT,bands[i,1],bands[i,2],m)*(z -> G(i,i,z+1im*ep)))
#             else
#                 tB[j,i] = 2*(DefiniteIntegral(transformW,bands[i,1],bands[i,2],m)*(z -> G(j,i,z+1im*ep)))
#             end
#         end
#     end

#     B = copy(tB)
#     B[:,1] = tB[:,1]
#     for j = 2:g
#        B[:,j] = B[:,j] + B[:,j-1]
#     end
#     B = 2im*pi*(A\B)  #Riemann Matrix

    	Ωx = zeros(Complex{Float64},g);
		for i = 1:g
			b = gaps[i,1]
			divfun = ff -> CircleFun(ff.a,ff.r,(ff.v./(circle_points(ff) .- b)).*sqrt.(circle_points(ff)))
			df = map(divfun,γs)
			Ωx[i] = 2*sum(map(Integrate,df))
		end
		#display(Ωx)
    	Ωx = A\Ωx

    	E = zeros(Complex{Float64},g);
		for i = 1:g
		    b = gaps[i,1]
		    divfun = ff -> CircleFun(ff.a,ff.r,(ff.v./(PeriodicKdV.circle_points(ff) .- b)).*PeriodicKdV.circle_points(ff))
		    df = map(divfun,γs)
		    E[i] = -0.5*sum(Ωx.*map(Integrate,df))
		end

    	Ωt = zeros(Complex{Float64},g);
		for i = 1:g
			b = gaps[i,1]
			divfun = ff -> CircleFun(ff.a,ff.r,(ff.v./(circle_points(ff) .- b)).*sqrt.(circle_points(ff)).^3)
			df = map(divfun,γs)
			Ωt[i] = 8*sum(map(Integrate,df))
		end
    	E += Ωt/8
    	E /= -2im*pi
    	Ωt = A\Ωt
	else
		for j = 1:g
			A[:,j] = map( k -> -2*Abelvec(m,k,gaps[j,1]), 1:g)
		end
		#display(A)

		Ωx = zeros(Complex{Float64},g);
		for j = 1:g
			Ωx += map( k -> -4*powervec(m,k,gaps[j,1],1), 1:g)
		end
		#display(Ωx)
    	Ωx = A\Ωx

		E = zeros(Complex{Float64},g);
		for j = 1:g
		    data = map( k -> -powervec(m,j,gaps[k,1],2), 1:g)
		    E[j] = -sum(Ωx.*data)
		end

		Ωt = zeros(Complex{Float64},g);
		for j = 1:g
			Ωt += map( k -> -16*powervec(m,k,gaps[j,1],3), 1:g)
		end
		E += Ωt/8
    	E /= -2im*pi
    	Ωt = A\Ωt

	end

	abel =  map( k -> -zs[1,2]*Abelvec(m,k,zs[1,1]), 1:g)
	for j = 2:g
		abel += map( k -> -zs[j,2]*Abelvec(m,k,zs[j,1]), 1:g)
	end
    abel = (A\abel)*(2*pi)

    HyperellipticSurface(gaps,g,zs,abel,Ωx,Ωt,E,α1)

end

struct BakerAkhiezerFunction
    WIm::Vector{WeightedInterval}
    WIp::Vector{WeightedInterval}
	bands::Array{Float64,2}
    Ω::Function
    E::Complex{Float64}
	F::Float64
    α1
    Cp # Compressed Cauchy matrix
    Cm
	nmat::Array{Int64}
    ns
	gridmat::Array{Vector{ComplexF64}}
    fftmat::Array{FFTW.r2rFFTWPlan}
    tol
    iter
end

Skew = θ -> [0 exp(θ); exp(-θ) 0]
Skewx = (θ,θx) -> [0 θx*exp(θ); -θx*exp(-θ) 0]
gp = (x,θ) -> Cut(Skew(1im*θ), x)
gm = (x,θ) -> Cut(Skew(-1im*θ), x)

## Swap definitions?
gpx = (x,θ,θx) -> Cut(Skewx(1im*θ,1im*θx), x)
gmx = (x,θ,θx) -> Cut(Skewx(-1im*θ,-1im*θx), x)


function bernsteinρ(a,b,r)
    rr = 2*r/(b-a) + 1.0
    ρ =  rr - sqrt(rr^2 - 1)
end

function bernsteinρ(v::Array) #tune rad/3
	if size(v)[1] == 1
		rad = 2*v[1,1]
	else
		rad = min(v[2,1]-v[1,2],2*v[1,1])
	end
    u = copy(v[:,1])
    u[1] = bernsteinρ(v[1,1],v[1,2],rad/2)
    for i = 2:length(u)-1
        rad = min(v[i,1]-v[i-1,2],v[i+1,1]-v[i,2])
        u[i] = bernsteinρ(v[i,1],v[i,2],rad/2)
    end
	if size(v)[1] > 1
    	rad = v[end,1]-v[end-1,2]
    	u[end] = bernsteinρ(v[end,1],v[end,2],rad/2)
	end
    u
end

fff = (ϵ,ρ,c,k) -> max(ρ < 1e-16 ? 0 : convert(Int64, (log(1/ϵ) + log(c/(1-ρ)))/log(1/ρ) |> ceil),k)

function choose_order(gaps::Array,ϵ,c,k)
    map(z -> fff(ϵ,z,c,k), bernsteinρ(gaps)) .+ 2
end

# function BakerAkhiezerFunction(S::HyperellipticSurface,n::Int64,tols = [2*1e-14,false],iter = 100)
#     zgaps_neg = hcat(- sqrt.(S.gaps[:,2]) |> reverse, - sqrt.(S.gaps[:,1]) |> reverse)
#     zgaps_pos = hcat( sqrt.(S.gaps[:,1]) , sqrt.(S.gaps[:,2]) )
#     #zzs_pos = sqrt.(zs)
#     #zzs_neg = -sqrt.(zs) |> reverse;
#     fV = (x,y) -> WeightedInterval(x,y,chebV)
#     fW = (x,y) -> WeightedInterval(x,y,chebW)
#     Ω = (x,t) -> S.Ωx*x + S.Ωt*t + S.Ω0
#
#     WIm = map(fW,zgaps_neg[:,1],zgaps_neg[:,2])
#     WIp = map(fV,zgaps_pos[:,1],zgaps_pos[:,2])
#
#     Ωs = Ω(0.0,0.0)
#     RHP = vcat(map(gm,WIm,Ωs |> reverse),map(gp,WIp,Ωs));
#     ns = fill(n,WIp |> length)
#
# #     #lens = abs.(zgaps[:,1] - zgaps[:,2])
#
#
# #     f = x -> convert(Int,ceil(10 + 10/x^2))
# #     ns = map(f,zgaps_pos[:,1])
#     ns = vcat(ns |> reverse, ns)
#
#     CpBO = CauchyChop(RHP,RHP,ns,ns,1,tols[2])
#     CmBO = CauchyChop(RHP,RHP,ns,ns,-1,tols[2])
#
#
#     #println("Effective rank of Cauchy operator = ",effectiverank(CpBO))
#     #println("Maximum rank of Cauchy operator = ", (2*S.g)^2*n )
#
#
#     return BakerAkhiezerFunction(WIm,WIp,Ω,S.E[1],S.α1,CpBO,CmBO,ns,tols[1],iter)
# end

M = (a,b) ->  (x -> (b-a)/2*(x .+ (b+a)/(b-a)))
iM = (a,b) -> (x -> 2/(b-a)*(x .- (b+a)/2))
J₊(z) = z-√(z-1 |> Complex)*√(z+1 |> Complex) #inverse Joukowsky map
gV(a,b) = z -> (1/2π)*(-1 + sqrt((z-a)/(z-b) |> Complex))*(2/(b-a))
gW(a,b) = z -> (1/2π)*(1 - sqrt((z-b)/(z-a) |> Complex))*(2/(b-a))

function BakerAkhiezerFunction(S::HyperellipticSurface,c::Float64;tols = [2*1e-14,1e-14],iter = 100,K=0,show_flag=false,choose_points = "adaptive",max_pts = 1000)
    zgaps_neg = hcat(- sqrt.(S.gaps[:,2]) |> reverse, - sqrt.(S.gaps[:,1]) |> reverse)
    zgaps_pos = hcat( sqrt.(S.gaps[:,1]) , sqrt.(S.gaps[:,2]) )
	bands = [zgaps_neg; zgaps_pos]
	F = sum(S.gaps[:,2] - S.gaps[:,1])
    #zzs_pos = sqrt.(zs)
    #zzs_neg = -sqrt.(zs) |> reverse;
    fV = (x,y) -> WeightedInterval(x,y,chebV)
    fW = (x,y) -> WeightedInterval(x,y,chebW)
    Ω = (x,t) -> S.Ωx*x + S.Ωt*t + S.Ω0

    WIm = map(fW,zgaps_neg[:,1],zgaps_neg[:,2])
    WIp = map(fV,zgaps_pos[:,1],zgaps_pos[:,2])

    Ωs = Ω(0.0,0.0)
    RHP = vcat(map(gm,WIm,Ωs |> reverse),map(gp,WIp,Ωs));
	if choose_points == "adaptive"
    	ns = map(x -> min(x,max_pts), choose_order(zgaps_pos,tols[1],c,K))
	elseif typeof(choose_points) <: Vector
		if length(choose_points) >= length(WIp)
			ns = choose_points[1:length(WIp)]
		else
			ns = vcat(choose_points,fill(choose_points[end],length(WIp)-length(choose_points)))
		end
	else
		ns = fill(choose_points,WIp |> length)
	end
    if show_flag
    	println(ns)
    end

	g = size(zgaps_neg,1)
	nv = zeros(Int64,2g,2g)
    for j = 1:g
        for k = 1:j-1
            aa = abs(gW(bands[j,1],bands[j,2])(bands[k,2]))
            jj = abs(J₊(iM(bands[j,1],bands[j,2])(bands[k,2])))
            val = ceil(log(jj,tols[2]/aa)) |> Int
            if val < 0
                val = 0
            end
            nv[j,k] = max(val,K)
        end
        for k = j+1:2g
            aa = abs(gW(bands[j,1],bands[j,2])(bands[k,1]))
            jj = abs(J₊(iM(bands[j,1],bands[j,2])(bands[k,1])))
            val = ceil(log(jj,tols[2]/aa)) |> Int
            if val < 0
                val = 0
            end
            nv[j,k] = max(val,K)
        end
        nv[j,j] = maximum(nv[j,:])
    end
    for j = g+1:2g
        for k = 1:j-1
            aa = abs(gV(bands[j,1],bands[j,2])(bands[k,2]))
            jj = abs(J₊(iM(bands[j,1],bands[j,2])(bands[k,2])))
            val = ceil(log(jj,tols[2]/aa)) |> Int
            if val < 0
                val = 0
            end
            nv[j,k] = max(val,K)
        end
        for k = j+1:2g
            aa = abs(gV(bands[j,1],bands[j,2])(bands[k,1]))
            jj = abs(J₊(iM(bands[j,1],bands[j,2])(bands[k,1])))
            val = ceil(log(jj,tols[2]/aa)) |> Int
            if val < 0
                val = 0
            end
            nv[j,k] = max(val,K)
        end
        nv[j,j] = maximum(nv[j,:])
    end

#     #lens = abs.(zgaps[:,1] - zgaps[:,2])


#     f = x -> convert(Int,ceil(10 + 10/x^2))
#     ns = map(f,zgaps_pos[:,1])
    ns = vcat(ns |> reverse, ns)
	#nv = kron(ns,ones(1,length(ns)))

    CpBO = CauchyChop(RHP,RHP,ns,ns,1,tols[2])
    CmBO = CauchyChop(RHP,RHP,ns,ns,-1,tols[2])


    #println("Effective rank of Cauchy operator = ",effectiverank(CpBO))
    #println("Maximum rank of Cauchy operator = ", (2*S.g)^2*n )

	gridmat = Array{Vector{ComplexF64}}(undef,2g) #store collocation points
    fftmat = Array{FFTW.r2rFFTWPlan}(undef,2g,2)
    for j = 1:2g
        gridmat[j] = M(bands[j,1],bands[j,2]).(Ugrid(nv[j,j])) .|> Complex
        fftmat[j,1] = FFTW.plan_r2r(zeros(ComplexF64,nv[j,j]),FFTW.REDFT11)
        fftmat[j,2] = FFTW.plan_r2r(zeros(ComplexF64,nv[j,j]),FFTW.RODFT11)
    end

    return BakerAkhiezerFunction(WIm,WIp,bands,Ω,S.E[1],F,S.α1,CpBO,CmBO,nv,ns,gridmat,fftmat,tols[1],iter)
end

function BakerAkhiezerFunction(S::HyperellipticSurface,c::Array;tols = [2*1e-14,false],iter = 100)
    zgaps_neg = hcat(- sqrt.(S.gaps[:,2]) |> reverse, - sqrt.(S.gaps[:,1]) |> reverse)
	bands = [zgaps_neg; zgaps_pos]
    zgaps_pos = hcat( sqrt.(S.gaps[:,1]) , sqrt.(S.gaps[:,2]) )
	F = sum(S.gaps[:,2] - S.gaps[:,1])
    #zzs_pos = sqrt.(zs)
    #zzs_neg = -sqrt.(zs) |> reverse;
    fV = (x,y) -> WeightedInterval(x,y,chebV)
    fW = (x,y) -> WeightedInterval(x,y,chebW)
    Ω = (x,t) -> S.Ωx*x + S.Ωt*t + S.Ω0

    WIm = map(fW,zgaps_neg[:,1],zgaps_neg[:,2])
    WIp = map(fV,zgaps_pos[:,1],zgaps_pos[:,2])

    Ωs = Ω(0.0,0.0)
    RHP = vcat(map(gm,WIm,Ωs |> reverse),map(gp,WIp,Ωs));
    ns = c # choose_order(zgaps_pos,tol,c)
#     #lens = abs.(zgaps[:,1] - zgaps[:,2])
#     f = x -> convert(Int,ceil(10 + 10/x^2))
#     ns = map(f,zgaps_pos[:,1])
    ns = vcat(ns |> reverse, ns)
	nv = kron(ns,ones(1,length(ns)))
    CpBO = CauchyChop(RHP,RHP,ns,ns,1,tols[2])
    CmBO = CauchyChop(RHP,RHP,ns,ns,-1,tols[2])
    #println("Effective rank of Cauchy operator = ",effectiverank(CpBO))
    #println("Maximum rank of Cauchy operator = ", (2*S.g)^2*n )

	g = size(zgaps_neg,1)
	gridmat = Array{Vector{ComplexF64}}(undef,2g) #store collocation points
    fftmat = Array{FFTW.r2rFFTWPlan}(undef,2g,2)
    for j = 1:2g
        gridmat[j] = M(bands[j,1],bands[j,2]).(Ugrid(ns[j])) .|> Complex
        fftmat[j,1] = FFTW.plan_r2r(zeros(ComplexF64,ns[j]),FFTW.REDFT11)
        fftmat[j,2] = FFTW.plan_r2r(zeros(ComplexF64,ns[j]),FFTW.RODFT11)
    end

    return BakerAkhiezerFunction(WIm,WIp,bands,Ω,S.E[1],F,S.α1,CpBO,CmBO,nv,ns,gridmat,fftmat,tols[1],iter)
end

function (BA::BakerAkhiezerFunction)(x,t,tol = BA.tol; directsolve = false, getmatrices = false)
    ns = BA.ns
    Ωs = BA.Ω(x,t)
    Ωsx = BA.Ω(1.0,0) - BA.Ω(0.0,0)

    RHP = vcat(map(gm,BA.WIm, Ωs |> reverse),map(gp,BA.WIp,Ωs));
    RHPx = vcat(map(gmx,BA.WIm, Ωs |> reverse, Ωsx |> reverse),map(gpx,BA.WIp,Ωs,Ωsx));

    m = length(RHP)
    #Z = ZeroOperator(n)
    #ZZ = BlockOperator(fill(Z,m,m))
    ZZ = BlockZeroOperator(ns)
    C⁺ = vcat(hcat(BA.Cp,ZZ),hcat(ZZ,BA.Cp))


    D11 = DiagonalBlockOperator( [-RHP[j].J[1,1] for j = 1:m], ns)
    D21 = DiagonalBlockOperator( [-RHP[j].J[1,2] for j = 1:m], ns)
    D12 = DiagonalBlockOperator( [-RHP[j].J[2,1] for j = 1:m], ns)
    D22 = DiagonalBlockOperator( [-RHP[j].J[2,2] for j = 1:m], ns)

    D11x = DiagonalBlockOperator( [-RHPx[j].J[1,1] for j = 1:m], ns)
    D21x = DiagonalBlockOperator( [-RHPx[j].J[1,2] for j = 1:m], ns)
    D12x = DiagonalBlockOperator( [-RHPx[j].J[2,1] for j = 1:m], ns)
    D22x = DiagonalBlockOperator( [-RHPx[j].J[2,2] for j = 1:m], ns)

    p = [1,m+1]
    for i = 2:m
        append!(p,i)
        append!(p,m+ i)
    end

    JC⁻ = vcat(hcat(D11*BA.Cm,D12*BA.Cm),hcat(D21*BA.Cm,D22*BA.Cm))
    JxC⁻ = vcat(hcat(D11x*BA.Cm,D12x*BA.Cm),hcat(D21x*BA.Cm,D22x*BA.Cm))
    S = C⁺ + JC⁻
    # notes for x derivative of solution
    # S*u = b
    # Sx*u + S*ux = bx
    # S*ux = bx - Sx*u,   Sx = JxC-

	ind = vcat(ns,ns)
	dim = sum(ind)
	ind = ind |> gapstoindex

	b = BlockVector(ind,fill(0.0im,dim))
	bx =  BlockVector(ind,fill(0.0im,dim))
	for j in 1:m
        b[j] = b[j] .+ (RHP[j].J[1,1] + RHP[j].J[2,1] - 1)
        b[j+m] = b[j+m] .+ (RHP[j].J[2,2] + RHP[j].J[1,2] - 1)
        bx[j] = bx[j] .+ (RHPx[j].J[1,1] + RHPx[j].J[2,1])
        bx[j+m] = bx[j+m] .+ (RHPx[j].J[2,2] + RHPx[j].J[1,2])
    end

	if directsolve
		Smat = S |> Array
		solvec = Smat\Vector(b)
		JxC⁻mat = JxC⁻ |> Array
		solvecx = Smat\(Vector(bx) - JxC⁻mat*solvec)
		sol = BlockVector(ind,solvec)
		solx = BlockVector(ind,solvecx)
	else
		Sp = permute(S,p)
	    JxCp = permute(JxC⁻,p)
		bp = permute(b,p)
	    bpx = permute(bx,p)

		D = TakeDiagonalBlocks(Sp,2)

		if getmatrices
			return (Array(Sp),Array(D))
		end

		for i = 1:size(Sp.A)[1]
			Sp[i,i] = ZeroOperator((Sp[i,i] |> size)...)
		end

		for i = 1:2:size(Sp.A)[1]-1
			Sp[i,i+1] = ZeroOperator((Sp[i,i] |> size)...)
			Sp[i+1,i] = ZeroOperator((Sp[i,i] |> size)...)
		end

    	fine_ind = indextogaps(ind)[p] |> gapstoindex
    	coarse_ind = fine_ind |> indextogaps
    	coarse_ind = coarse_ind[1:2:end] + coarse_ind[2:2:end] |> gapstoindex

    	PrS = x -> BlockVector(fine_ind,D\BlockVector(coarse_ind,x))
		Op = x -> x + PrS(Sp*x)
		#Opmat = Op |> Array
		#e1 = zeros(size(Vector(bp)))
		#=f1 = BlockVector(ind,fill(0.0im,dim))
		e1 = permute(f1,p)
		e1[1][1] = 1.
		#println(norm(Opmat*e1))
		println(sqrt(abs(Op(e1)⋅Op(e1))))=#
    	out = GMRES_quiet(Op,PrS(bp),⋅, tol,BA.iter)
    	solp = out[2][1]*out[1][1]
    	for j = 2:length(out[2])
        	solp += out[2][j]*out[1][j]
    	end

    	outx = GMRES_quiet(Op,PrS(bpx-JxCp*solp),⋅, tol,BA.iter)
    	solpx = outx[2][1]*outx[1][1]
    	for j = 2:length(outx[2])
        	solpx += outx[2][j]*outx[1][j]
    	end

		sol = ipermute(solp,p)
	    solx = ipermute(solpx,p)

    end

	f1 = [WeightFun(sol[j],RHP[j].W)  for j = 1:m ]
    f2 = [WeightFun(sol[m + j],RHP[j].W)  for j = 1:m ]
    f1x = [WeightFun(solx[j],RHP[j].W)  for j = 1:m ]
    f2x = [WeightFun(solx[m + j],RHP[j].W)  for j = 1:m ]

    Φ = (z,o) -> [Cauchy(f1,z,o) + 1.0, Cauchy(f2,z,o) + 1.0] |> transpose
	Φx = (z,o) -> [Cauchy(f1x,z,o), Cauchy(f2x,z,o)] |> transpose
    (Φ,Φx,f1,f2,f1x,f2x,BA.E) #might wanna change back eventually
    #(Φ,f1,f2)
end

function KdV(BA::BakerAkhiezerFunction,x,t; directsolve = false)
    out = BA(x+6*BA.α1*t, t; directsolve);
    1/pi*sum(map(x -> DomainIntegrateVW(x), out[5])) + 2*BA.E - BA.α1 #change index back eventually
    # I think this is right but I cannot justify it, +/- sign issue
    # It must have to do with the jumps and which sheet, etc.
end

function KdV(BA::BakerAkhiezerFunction,x,t,tol; directsolve = false)
    out = BA(x+6*BA.α1*t, t, tol; directsolve);
    1/pi*sum(map(x -> DomainIntegrateVW(x), out[5])) + 2*BA.E - BA.α1 #change index back eventually
    # I think this is right but I cannot justify it, +/- sign issue
    # It must have to do with the jumps and which sheet, etc.
end
