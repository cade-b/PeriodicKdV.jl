M = (a,b) ->  (x -> (b-a)/2*x .+ (b+a)/2)
iM = (a,b) -> (x -> 2/(b-a)*(x .- (b+a)/2))
eps = 1e-15

struct Weight
    a
    b
    seed
    w
end

struct WeightedInterval
    a::Float64
    b::Float64
    W::Weight
end

struct Cut
    J::Array
    W::WeightedInterval
end

function ==(W1::WeightedInterval,W2::WeightedInterval)
    y = rand()
    (W1.a == W2.a) && (W1.b == W2.b) && (W1.W.w(y) == W2.W.w(y) )
end

struct WeightFun
    cs::Vector
    W::WeightedInterval
end

InInterval(x,WI) = WI.a <= x <= WI.b

function (F::WeightFun)(x::Float64)
    if InInterval(x,F.W)
        (poly(F.W.W.a,F.W.W.b,length(F.cs)-1,[iM(F.W.a,F.W.b)(x)])*F.cs)[1]
    else
        return 0.0
    end
end

function (F::WeightFun)(x::Vector{Float64})
    poly(F.W.W.a,F.W.W.b,length(F.cs)-1,iM(F.W.a,F.W.b)(x))*F.cs
end

function WeightPlot(C::WeightedInterval)
    x = -1:0.01:1
    a = W.a
    b = W.b
    X = map(M(a,b),x)
    p = plot(X,0*X,lw = 4, legend = false, yrange = [-1,2], xrange = [a-1,b+1], color=:blue)
    plot!([a,b],[0,0], seriestype = :scatter, markersize = 10,
    markercolor = :black)
    plot!(X,map(W.W.w,x), color=:black)
end

function WeightPlot(WA::Array)
    x = -1:0.01:1
    W = WA[1]
    a = W.a
    b = W.b
    aa = minimum(map(x -> x.a,WA))
    bb = maximum(map(x -> x.b,WA))
    X = map(M(a,b),x)
    p = plot(X,0*X,lw = 4, legend = false, yrange = [-1,2], xrange = [aa-1,bb+1],color = :blue)
    plot!([a,b],[0,0], seriestype = :scatter, markersize = 5,
    markercolor = :black)
    plot!(X,map(W.W.w,x), color=:black)
    for i = 2:length(WA)
        W = WA[i]
        a = W.a
        b = W.b
        X = map(M(a,b),x)
        p = plot!(X,0*X,lw = 4, legend = false, yrange = [-1,2], xrange = [aa-1,bb+1],color = :blue)
        plot!([a,b],[0,0], seriestype = :scatter, markersize = 5,
        markercolor = :black)
        plot!(X,map(W.W.w,x), color=:black)
    end
    p
end

function FunPlot(FA::Array)
    x = -1:0.01:1 |> Array
    F = FA[1]
    W = F.W
    a = W.a
    b = W.b
    aa = minimum(map(x -> x.W.a,FA))
    bb = maximum(map(x -> x.W.b,FA))
    X = map(M(a,b),x)
    p = plot(X,0*X,lw = 4, legend = false, yrange = [-3,3], xrange = [aa-1,bb+1],color = :blue)
    plot!([a,b],[0,0], seriestype = :scatter, markersize = 5,
    markercolor = :black)

    data = poly(W.W.a,W.W.b,length(F.cs)-1,x)*F.cs
    plot!(X,data |> real, color=:blue)
    plot!(X,data |> imag, color=:red)
    plot!(X,map(W.W.w,x), color=:black)

    for i = 2:length(FA)
        F = FA[i]
        W = F.W
        a = W.a
        b = W.b
        X = map(M(a,b),x)
        p = plot!(X,0*X,lw = 4, legend = false, yrange = [-5,5], xrange = [aa-1,bb+1],color = :blue)
        plot!([a,b],[0,0], seriestype = :scatter, markersize = 5,
        markercolor = :black)

        data = poly(W.W.a,W.W.b,length(F.cs)-1,x)*F.cs
        plot!(X,data |> real, color=:blue)
        plot!(X,data |> imag, color=:red)
        plot!(X,map(W.W.w,x), color=:black)
    end
    p

end

# TODO: this should be more flexible and robust
# Need to create structs <: WeightedInterval for each Chebyshev type
function DomainIntegrateVW(f::WeightFun)
    (f.W.b -  f.W.a)/2*f.cs[1]
end
