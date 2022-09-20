# Problem Set 3
    # Collin DeVore
        # with Bowei Dong and Will Myers

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables

function ps3()
# Question 1    
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS3-gev/nlsw88w.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation

    function mlogit(optvector, X, y)
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N, J)
        alpha = optvector[1:end-1]
        gamma = optvector[end]
        bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
        num = zeros(N, J)
        dem = zeros(N)
        for j = 1:J
            bigY[:,j] = y.==j
            diffZ = Z[:,j] - Z[:,J]
            num[:,j] = exp.((X*bigAlpha[:,j]) .+ (gamma*diffZ))
            dem .+= num[:,j]
        end
        P = num./repeat(dem, 1, J)
        loglike = -sum(bigY .* log.(P))
        return loglike
    end

    alpha_zero = zeros(7 * size(X, 2) + 1)
    alpha_rand = rand(7 * size(X, 2) + 1)
    alpha_hat_optim = optimize(optvector -> mlogit(optvector, X, y), alpha_rand, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100000, show_trace = true, show_every = 50))
    alpha_hat_mle = alpha_hat_optim.minimizer
    println(alpha_hat_mle)



# Question 2    
#=
gamma is the impact of the wages (diffZ) on the probability of the occupation.
=#



# Question 3    
    function nlogit(optvector2, X, Z, y)
        for j = 1:J
            diffZ = Z[:,j] - Z[:,J]
        end
        ZNest1 = hcat[diffZ.elnwage1 diffZ.elnwage2 diffZ.elnwage3]
        ZNest2 = hcat[diffZ.elnwage4 diffZ.elnwage5 diffZ.elnwage6 diffZ.elnwage7]
        ZNest3 = hcat[diffZ.elnwage8]
        alpha = optvector2[1:end-1]
        gamma = optvector2[end]
        denom =(exp(((X .* alphaWC) .+ (diffZ.elnwage1 .* gamma)) ./ lambdaWC) .+ exp(((X .* alphaWC) .+ (diffZ.elnwage2 .* gamma)) ./ lambdaWC) .+ exp(((X .* alphaWC) .+ (diffZ.elnwage3 .* gamma)) ./ lambdaWC))^lambdaWC .+ (exp(((X .* alphaBC) .+ (diffZ.elnwage4 .* gamma)) ./ lambdaBC) .+ exp(((X .* alphaBC) .+ (diffZ.elnwage5 .* gamma)) ./ lambdaBC) .+ exp(((X .* alphaBC) .+ (diffZ.elnwage6 .* gamma)) ./ lambdaBC) .+ exp(((X .* alphaBC) .+ (diffZ.elnwage7 .* gamma)) ./ lambdaBC))^lambdaBC .+ exp((X .* alphaO) .+ (diffZ.elnwage8 .* gamma))
        pprof = (exp(((X .* alphaWC) .+ (diffZ.elnwage1 .* gamma)) ./ lambdaWC) .* ((exp(((X .* alphaWC) .+ (diffZ.elnwage1 .* gamma)) ./ lambdaWC) .+ exp(((X .* alphaWC) .+ (diffZ.elnwage2 .* gamma)) ./ lambdaWC) .+ exp(((X .* alphaWC) .+ (diffZ.elnwage3 .* gamma)) ./ lambdaWC)) .^ (1 - lambdaWC))) ./ denom
        pmanage = (exp(((X .* alphaWC) .+ (diffZ.elnwage2 .* gamma)) ./ lambdaWC) .* ((exp(((X .* alphaWC) .+ (diffZ.elnwage1 .* gamma)) ./ lambdaWC) .+ exp(((X .* alphaWC) .+ (diffZ.elnwage2 .* gamma)) ./ lambdaWC) .+ exp(((X .* alphaWC) .+ (diffZ.elnwage3 .* gamma)) ./ lambdaWC)) .^ (1 - lambdaWC))) ./ denom
        psales = (exp(((X .* alphaWC) .+ (diffZ.elnwage3 .* gamma)) ./ lambdaWC) .* ((exp(((X .* alphaWC) .+ (diffZ.elnwage1 .* gamma)) ./ lambdaWC) .+ exp(((X .* alphaWC) .+ (diffZ.elnwage2 .* gamma)) ./ lambdaWC) .+ exp(((X .* alphaWC) .+ (diffZ.elnwage3 .* gamma)) ./ lambdaWC)) .^ (1 - lambdaWC))) ./ denom
        pcleric = (exp(((X .* alphaBC) .+ (diffZ.elnwage4 .* gamma)) ./ lambdaBC) .* ((exp(((X .* alphaBC) .+ (diffZ.elnwage4 .* gamma)) ./ lambdaBC) .+ exp(((X .* alphaBC) .+ (diffZ.elnwage5 .* gamma)) ./ lambdaBC) .+ exp(((X .* alphaBC) .+ (diffZ.elnwage6 .* gamma)) ./ lambdaBC) .+ exp(((X .* alphaBC) .+ (diffZ.elnwage7 .* gamma)) ./ lambdaBC)) .^ (1 - lambdaBC))) ./ denom
        pcrafts = (exp(((X .* alphaBC) .+ (diffZ.elnwage5 .* gamma)) ./ lambdaBC) .* ((exp(((X .* alphaBC) .+ (diffZ.elnwage4 .* gamma)) ./ lambdaBC) .+ exp(((X .* alphaBC) .+ (diffZ.elnwage5 .* gamma)) ./ lambdaBC) .+ exp(((X .* alphaBC) .+ (diffZ.elnwage6 .* gamma)) ./ lambdaBC) .+ exp(((X .* alphaBC) .+ (diffZ.elnwage7 .* gamma)) ./ lambdaBC)) .^ (1 - lambdaBC))) ./ denom
        poperat = (exp(((X .* alphaBC) .+ (diffZ.elnwage6 .* gamma)) ./ lambdaBC) .* ((exp(((X .* alphaBC) .+ (diffZ.elnwage4 .* gamma)) ./ lambdaBC) .+ exp(((X .* alphaBC) .+ (diffZ.elnwage5 .* gamma)) ./ lambdaBC) .+ exp(((X .* alphaBC) .+ (diffZ.elnwage6 .* gamma)) ./ lambdaBC) .+ exp(((X .* alphaBC) .+ (diffZ.elnwage7 .* gamma)) ./ lambdaBC)) .^ (1 - lambdaBC))) ./ denom
        ptransport = (exp(((X .* alphaBC) .+ (diffZ.elnwage7 .* gamma)) ./ lambdaBC) .* ((exp(((X .* alphaBC) .+ (diffZ.elnwage4 .* gamma)) ./ lambdaBC) .+ exp(((X .* alphaBC) .+ (diffZ.elnwage5 .* gamma)) ./ lambdaBC) .+ exp(((X .* alphaBC) .+ (diffZ.elnwage6 .* gamma)) ./ lambdaBC) .+ exp(((X .* alphaBC) .+ (diffZ.elnwage7 .* gamma)) ./ lambdaBC)) .^ (1 - lambdaBC))) ./ deonom
        pO = exp((X .* alphaO) .+ (diffZ.elnwage8 .* gamma)) ./ denom
        p = vcat[pprof; pmanage; psales; pcleric; pcrafts; poperat; ptransport; pO]
        loglik = sum(y .* log.(p) .+ ((1 .- y) .* log.(1 .- p)))
        return loglik
        optvector2 = vcat[alphaWC; alphaBC; alphaO; lambdaWC; lambdaBC; gamma]
    end

# alphahatloglik = optimize(optvector2 -> -logit(optvector2, X, Z, y), rand(size(X, 2)), LBFGS(), Optim.Options(g_tol = 1e-6), iterations = 100000, show_trace = true)
# println(alphahatloglik.minimizer)

    alpha_zero = zeros(7 * size(X, 2) + 1)
    alpha_rand = rand(7 * size(X, 2) + 1)
    alpha_hat_optim2 = optimize(optvector2 -> mlogit(optvector2, X, y), alpha_rand, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100000, show_trace = true, show_every = 50))
    alpha_hat_mle2 = alpha_hat_optim2.minimizer
    println(alpha_hat_mle2)
end
ps3()


































