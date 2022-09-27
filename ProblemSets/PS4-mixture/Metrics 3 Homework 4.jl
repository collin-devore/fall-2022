# Problem Set 3
    # Collin DeVore
        # with Bowei Dong and Will Myers

using Distributions, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables

function ps4()
# Question 1
    # Code used as a mixture of my Metrics 3 Homework 3 code and Dr. Ransom's ps3solutions.jl code
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS3-gev/nlsw88w.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation

    function mlogit(optvector, X, Z, y)
        alpha = optvector[1:end-1]
        gamma = optvector[end]
        K = size(X, 2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N, J)
        bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
        T = promote_type(eltype(X), eltype(optvector))
        num = zeros(T, N, J)
        dem = zeros(T, N)
        for j = 1:J
            bigY[:, j] = y.==j
            num[:, j] = exp.(X * bigAlpha[:, j] .+ (Z[:, j] .- Z[:, J]) * gamma)
            dem .+= num[:, j]
        end
        P = num./repeat(dem, 1, J)
        loglike = -sum(bigY .* log.(P))
        return loglike
    end

    # Estimated Values from PS3 Q1
    alpha_zero = zeros(7 * size(X, 2) + 1)
    alpha_rand = rand(7 * size(X, 2) + 1)
    alpha_hat_optim = optimize(optvector -> mlogit(optvector, X, Z, y), alpha_rand, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100000, show_trace = true, show_every = 50))
    alpha_hat_mle = alpha_hat_optim.minimizer
    println(alpha_hat_mle)

    # Automatic Differentiation Using alpha_hat_mle as the Starting Values
    startvals = alpha_hat_mle
    td = TwiceDifferentiable(optvector -> mlogit(optvector, X, Z, y), startvals; autodiff = :forward)
    optvector_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100000, show_trace = true, show_every = 50))
    optvector_mle_ad = optvector_optim_ad.minimizer
    println(optvector_mle_ad)

    # Standard Errors
    H = Optim.hessian!(td, optvector_mle_ad)
    optvector_mle_ad_se = sqrt.(diag(inv(H)))
    println([optvector_mle_ad optvector_mle_ad_se])



# Question 2
#=
I do not think that gamma necessarily makes more sense than the gamma in Problem Set 3. For some reason, it looks like I am getting the same answer
for both, so it seems that they make equal sense for the values that I am getting.
=#



# Question 3
    # A
    include("lgwt.jl")
    d = Normal(0, 1)
    nodes, weights = lgwt(7, -4, 4)
    sum(weights .* pdf.(d, nodes))                      # 0.9545012573546632         Almost 1 (Integral of the pdf)
    sum(weights .* nodes .* pdf.(d, nodes))             # -2.7755575615628914e-17    Almost 0 (Mean)

    # B
        # Second Quadrature Practice
    d = Normal(0, 2)
    nodes, weights = lgwt(7, -10, 10)
    sum(weights .* nodes.^2 .* pdf.(d, nodes))          # 3.265514281891983          Almost 4 (Variance)
    
        # Third Quadrature Practice
    d = Normal(0, 2)
    nodes, weights = lgwt(10, -10, 10)
    sum(weights .* nodes.^2 .* pdf.(d, nodes))          # 4.038977384853661          Almost 4 (Variance)

#=
The quadrature does a pretty good job estimating the true variance of 4, especially when more quadrature points are used. More quadrature
points seems to allow the approximation to get closer to the correct value, since 4.04 with ten quadrature points is much closer than 3.27
estimated with seven quadrature points.
=#

    # C
        # Variance where D = 1000000
    Random.seed!(1234)
    X = rand(Normal(0, 2), 1000000)
    sum((10 + 10) * (1/1000000) * (X.^2) * (1/(10 + 10)))       # 4.0020052937510755        Almost 4 (Variance)

        # Mean where D = 1000000
    sum((10 + 10) * (1/1000000) * X * (1/(10 + 10)))            # -0.0007439473265286919    Almost 0 (Mean)

        # Integral of pdf where D = 1000000
    sum((10 + 10) * (1/1000000) * 1000000 * (1/(10 + 10)))      # 1                         1 (Integral of pdf)

        # Comment on fit when D = 1000 vs when D = 1000000
    Random.seed!(1111)
    X = rand(Normal(0, 2), 1000)
    sum((10 + 10) * (1/1000) * (X.^2) * (1/(10 + 10)))          # 3.772019540804709         Almost 4 (Variance)
    sum((10 + 10) * (1/1000) * X * (1/(10 + 10)))               # 0.09771575026974061       Almost 0 (Mean)
    sum((10 + 10) * (1/1000) * 1000 * (1/(10 + 10)))            # 1                         1 (Integral of pdf)

#=
When D = 1,000 the simulated integral fits the data pretty well, but not nearly as well as when D = 1,000,000. This is illustrated
by the fact that -0.00074 is much closer than 0.09772 to 0 (the mean), and 4.00201 is much closer than 3.77202 to 4 (the variance).
This makes sense because more observations should be closer to the true value by the law of large numbers.
=#

    # D
#=
These two are extremely similar. They are exactly equal if the weight is equal to (b-a)/D and ξ is equal to X.
=#



# Question 4
    # With help from Dr. Ransom in office hours 9/26/2022, his suggested code used and adjusted
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS3-gev/nlsw88w.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation


    function mlogit(optvector, X, Z, y, omega, xi, mu, sigma)
        alpha = optvector[1:end-1]
        gamma = optvector[end]
        K = size(X, 2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N, J)
        bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
        T = promote_type(eltype(X), eltype(optvector))
        num = zeros(T, N, J)
        dem = zeros(T, N)
        for j = 1:J
            bigY[:, j] = y.==j
            num[:, j] = exp.(X * bigAlpha[:, j] .+ (Z[:, j] .- Z[:, J]) * gamma)
            dem .+= num[:, j]
        end
        P = num./repeat(dem, 1, J)
        function Pintegrated(optvector, X, Z, y, omega, xi, mu, sigma)
            integral1 = zeros(N*T, 1)
            for g=1:7
                Pkernl = pmlogit(optvector, X, Z, xi[g])
                Pkern2 = prod(Pkernl.^bigY, dims = 2)
                integral1 .+= omega[g] .* Pkern2 .* pdf(d, xi[g])
            end
        end
        loglike = -sum(bigY .* log.(Pintegrated(optvector, X, Z, y, omega, xi, mu, sigma)))
        return loglike
    end


#=
            # Test 2
    function pmlogit(optvector, X, Z, y, omega, xi, mu, sigma)    
        function Pintegrated(optvector, X, Z, y, omega, xi, mu, sigma)
            integral1 = zeros(N*T, 1)
            for g=1:7
                Pkernl = pmlogit(optvector, X, Z, xi[g])
                Pkern2 = prod(Pkernl.^bigY, dims = 2)
                integral1 .+= omega[g] .* Pkern2 .* pdf(d, xi[g])
            end
        end
        function mlogit(optvector, X, Z, y, omega, xi, mu, sigma)
            alpha = optvector[1:end-1]
            gamma = optvector[end]
            K = size(X, 2)
            J = length(unique(y))
            N = length(y)
            bigY = zeros(N, J)
            bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
            T = promote_type(eltype(X), eltype(optvector))
            num = zeros(T, N, J)
            dem = zeros(T, N)
            for j = 1:J
                bigY[:, j] = y.==j
                num[:, j] = exp.(X * bigAlpha[:, j] .+ (Z[:, j] .- Z[:, J]) * gamma)
                dem .+= num[:, j]
            end
            P = num./repeat(dem, 1, J)
        end
        loglike = -sum(bigY .* log.(Pintegrated(optvector, X, Z, y, omega, xi, mu, sigma)))
        return loglike
    end
=#


#=
    # Test 3
    function mlogit(optvector, X, Z, y, omega, xi, mu, sigma)
        alpha = optvector[1:end-1]
        gamma = optvector[end]
        K = size(X, 2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N, J)
        bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
        T = promote_type(eltype(X), eltype(optvector))
        num = zeros(T, N, J)
        dem = zeros(T, N)
        for j = 1:J
            bigY[:, j] = y.==j
            num[:, j] = exp.(X * bigAlpha[:, j] .+ (Z[:, j] .- Z[:, J]) * gamma)
            dem .+= num[:, j]
        end
        P = num./repeat(dem, 1, J)
        function Pintegrated(optvector, X, Z, y, omega, xi, mu, sigma)
            integral1 = zeros(N*T, 1)
            for g=1:7
                Pkernl = mlogit(optvector, X, Z, xi[g])
                Pkern2 = prod(Pkernl.^bigY, dims = 2)
                integral1 .+= omega[g] .* Pkern2 .* pdf(d, xi[g])
            end
        end
        loglike = -sum(bigY .* log.(Pintegrated(optvector, X, Z, y, omega, xi, mu, sigma)))
        return loglike
    end                     # Self-referencing issue?
=#    

# mlogit(7, X, Z, y, 3, 0, 1) 
# pintegrated(7, X, Z, y, 3, 0, 1)      
# pmlogit(7, X, Z, y, 3, 0, 1)
# Cannot get this to run for some reason. It seems like the issue is the pmlogit. It does not have a function defined
# If I define a function, that seems to make the whole mlogit function either self referential or inside out.


    # Optimizer (Same as in ps3 solutions)
    startvals = [2 * rand(7 * size(X, 2)) .- 1; .1]
    td = TwiceDifferentiable(optvector -> mlogit(optvector, X, Z, y), startvals; autodiff = :forward)
    optvector_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100000, show_trace = true, show_every = 50))
    optvector_mle_ad = optvector_optim_ad.minimizer
    println(optvector_mle_ad)


# Question 5
    # Based off of code for question 4 shown in Dr. Ransom's office hours, 9/26/2022
    function mlogit(optvector, X, Z, y, a, b, D, xi)
        alpha = optvector[1:end-1]
        gamma = optvector[end]
        K = size(X, 2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N, J)
        bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
        T = promote_type(eltype(X), eltype(optvector))
        num = zeros(T, N, J)
        dem = zeros(T, N)
        for j = 1:J
            bigY[:, j] = y.==j
            num[:, j] = exp.(X * bigAlpha[:, j] .+ (Z[:, j] .- Z[:, J]) * gamma)
            dem .+= num[:, j]
        end
        P = num./repeat(dem, 1, J)
        function Pintegrated(optvector, X, Z, y, a, b, D, xi)
            omega = (b-a)/D
            integral1 = zeros(N*T, 1)
            dens = 1/(b-a)
            rand(uniform(a, b, D))
            function pmlogit(optvector, X, Z, xi)
                P = num./repeat(dem, 1, J)
                likelihood = sum(bigY .* log.P)
            end
            for g=1:7
                Pkernl = pmlogit(optvector, X, Z, xi[g])
                Pkern2 = prod(Pkernl.^bigY, dims = 2)
                integral1 .+= omega[g] .* Pkern2 .* dens
            end
        end
        loglike = -sum(bigY .* log.(Pintegrated(optvector, X, Z, y, omega, xi, mu, sigma)))
        return loglike
    end


# Optimizer (Same as in ps3 solutions)
    startvals = [2 * rand(7 * size(X, 2)) .- 1; .1]
    td = TwiceDifferentiable(optvector -> mlogit(optvector, X, Z, y), startvals; autodiff = :forward)
    optvector_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100000, show_trace = true, show_every = 50))
    optvector_mle_ad = optvector_optim_ad.minimizer
    println(optvector_mle_ad)   
end
ps4()









