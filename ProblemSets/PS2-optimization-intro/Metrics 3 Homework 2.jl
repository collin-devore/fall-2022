# Homework 2
    # Collin DeVore
        # With Bowei Dong and Will Myers

# Question 1
    using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables

function ps2()
    f(x)= -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2
    negf(x) = x[1]^4 + 10x[1]^3 + 2x[1]^2 + 3x[1] + 2
    startval = rand(1)
    result = optimize(negf, startval, LBFGS())
    return result
    println(result)



# Question 2
    url = "https://raw.githubusercontent.com/OU-PHD-ECONOMETRICS/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    data = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(data,1),1) data.age data.race.==1 data.collgrad.==1]
    y = data.married.==1

    function ols(beta, X, y)
        ssr = (y.-X*beta)'*(y.-X*beta)
        return ssr
    end

    betahatols = optimize(b -> ols(b, X, y), rand(size(X, 2)), LBFGS(), Optim.Options(g_tol = 1e-6, iterations = 100000, show_trace = true))
    println(betahatols.minimizer)

    bols = inv(X'*X)*X'*y  # Same as betahatols.minimizer
    println(bols)

    data.white = data.race.==1
    bols_lm = lm(@formula(married ~ age + white + collgrad), data)  # Returns the same as betahatols.minimizer
    println(bols_lm)



# Question 3
    function logit(alpha, X, y)
        p1 = exp.(X * alpha) ./ (1 .+ exp.(X * alpha))
        loglik = sum(y .* log.(p1) .+ (1 .- y) .* log.(1 .- p1))
        return loglik
    end

    alphahatloglik = optimize(a -> -logit(a, X, y), rand(size(X, 2)), LBFGS(), Optim.Options(g_tol = 1e-6, iterations = 100000, show_trace = true))
    println(alphahatloglik.minimizer)



# Question 4
    bloglik_glm = glm(@formula(married ~ age + white + collgrad), data, Binomial(), LogitLink())
    printlm(bloglik_glm)



# Question 5
    freqtable(data, :occupation)
    data = dropmissing(data, :occupation)
    data[data.occupation.==8, :occupation] .= 7
    data[data.occupation.==9, :occupation] .= 7
    data[data.occupation.==10, :occupation] .= 7
    data[data.occupation.==11, :occupation] .= 7
    data[data.occupation.==12, :occupation] .= 7
    data[data.occupation.==13, :occupation] .= 7
    freqtable(data, :occupation)

    X = [ones(size(data, 1), 1) data.age data.race.==1 data.collgrad.==1]
    y = data.occupation

#=
function mlogit(alpha, X, y)
    for j = 1:length(X,2)
        for i = 1:length(X,1)
            P[i,j] = exp(X[i,j]*alpha[i,1])/(1 + exp(sum(X[i,j]*alpha[i,1])))
            loglike = y[i,j]*log(P[i,j])
        end
    end
    return loglike
end
=#

    function mlogit(alpha, X, y)
        alpha1 = reshape(alpha[1:4, 1], (4, 1))
        alpha2 = reshape(alpha[5:8, 1], (4, 1))
        alpha3 = reshape(alpha[9:12, 1], (4, 1))
        alpha4 = reshape(alpha[13:16, 1], (4, 1))
        alpha5 = reshape(alpha[17:20, 1], (4, 1))
        alpha6 = reshape(alpha[21:24, 1], (4, 1))
        p1 = exp.(X * alpha1) ./ sum(1 .+ exp.(X * alpha1) .+ exp.(X * alpha2) .+ exp.(X * alpha3) .+ exp.(X * alpha4) .+ exp.(X * alpha5) .+ exp.(X * alpha6))
        p2 = exp.(X * alpha2) ./ sum(1 .+ exp.(X * alpha1) .+ exp.(X * alpha2) .+ exp.(X * alpha3) .+ exp.(X * alpha4) .+ exp.(X * alpha5) .+ exp.(X * alpha6))
        p3 = exp.(X * alpha3) ./ sum(1 .+ exp.(X * alpha1) .+ exp.(X * alpha2) .+ exp.(X * alpha3) .+ exp.(X * alpha4) .+ exp.(X * alpha5) .+ exp.(X * alpha6))
        p4 = exp.(X * alpha4) ./ sum(1 .+ exp.(X * alpha1) .+ exp.(X * alpha2) .+ exp.(X * alpha3) .+ exp.(X * alpha4) .+ exp.(X * alpha5) .+ exp.(X * alpha6))
        p5 = exp.(X * alpha5) ./ sum(1 .+ exp.(X * alpha1) .+ exp.(X * alpha2) .+ exp.(X * alpha3) .+ exp.(X * alpha4) .+ exp.(X * alpha5) .+ exp.(X * alpha6))
        p6 = exp.(X * alpha6) ./ sum(1 .+ exp.(X * alpha1) .+ exp.(X * alpha2) .+ exp.(X * alpha3) .+ exp.(X * alpha4) .+ exp.(X * alpha5) .+ exp.(X * alpha6))
    
        d = zeros(size(X, 1), 6)
        for i in 1:size(d, 1)
            if y[i,1] == 1
                d[i,1] = 1
            elseif y[i, 1] == 2
                d[i,2] = 1
            elseif y[i, 1] == 3
                d[i,3] = 1
            elseif y[i, 1] == 4
                d[i,4] = 1
            elseif y[i, 1] == 5
                d[i,5] = 1
            else d[i,6] = 1
            end
        end
        loglike = sum((d[:,1] .* log.(p1)) .+ (d[:,2] .* log.(p2)) .+ (d[:,3] .* log.(p3)) .+ (d[:,4] .* log.(p4)) .+ (d[:,5] .* log.(p5)) .+ (d[:,6] .* log.(p6)))
        return loglike
    end

    alphahatloglike = optimize(a -> -mlogit(a, X, y), zeros(24, 1), LBFGS(), Optim.Options(g_tol = 1e-6, iterations = 100000, show_trace = true))
    println(alphahatloglike.minimizer)
end
ps2()





