# Problem Set 3
    # Collin DeVore
        # with Bowei Dong

using SMM, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV, FreqTables

function ps7()
# Question 1 (With help from slide 10 and lecture 2 answers)
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df, 1), 1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1

    function logit_gmm(alpha, X, y)
        P = X * alpha
        g = y .- P
        J = g'*I*g
        return J
    end

    alpha_hat_optim = optimize(a -> logit_gmm(a, X, y), rand(size(X, 2)), LBFGS(), Optim.Options(g_tol = 1e-8, iterations = 100000))
    println(alpha_hat_optim)



# Question 2
    # Part A (From Dr. Ransom's previous answer key)
    freqtable(df, :occupation)
    df = dropmissing(df, :occupation)
    df[df.occupation .== 8, :occupation] .= 7
    df[df.occupation .== 9, :occupation] .= 7
    df[df.occupation .== 10, :occupation] .= 7
    df[df.occupation .== 11, :occupation] .= 7
    df[df.occupation .== 12, :occupation] .= 7
    df[df.occupation .== 13, :occupation] .= 7
    freqtable(df, :occupation)

    X = [ones(size(df, 1), 1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation

    function mlogit(alpha, X, y)
        K = size(X, 2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N, J)
        bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
        num = zeros(N, J)
        dem = zeros(N)
        for j in 1:J
            bigY[:, j] = y .== j
            num[:, j] = exp.(X*bigAlpha[:, j])
            dem .+= num[:, j]
        end
        P = num./repeat(dem, 1, J)
        loglik = -sum(bigY .* log.(P))
        return(loglik)
    end

    alpha_rand = rand(6 * size(X, 2))
    alpha_hat_optim = optimize(a -> mlogit(a, X, y), alpha_rand, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100000, show_trace = true, show_every = 50))
    println(alpha_hat_optim.minimizer)


    # Part B
    function mlogit_gmm(alpha, X, y)
        K = size(X, 2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N, J)
        bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
        num = zeros(N, J)
        dem = zeros(N)
        for j in 1:J
            bigY[:, j] = y .== j
            num[:, j] = exp.(X*bigAlpha[:, j])
            dem .+= num[:, j]
        end
        P = num./repeat(dem, 1, J)
        g = bigY[:] .- P[:]
        J = g' * I * g
        return J
    end

    start_vals = alpha_hat_optim.minimizer
    alpha_hat_optim_gmm = optimize(a -> mlogit_gmm(a, X, y), start_vals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100000, show_trace = true, show_every = 50))
    println(alpha_hat_optim_gmm.minimizer)


    # Part C
    start_vals_2 = rand(6 * size(X, 2))
    alpha_hat_optim_gmm_2 = optimize(a -> mlogit_gmm(a, X, y), start_vals_2, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100000, show_trace = true, show_every = 50))
    println(alpha_hat_optim_gmm_2.minimizer)

#=
It seems reasonable right now to conclude that the estimates from b and c are globally concave. The coeffecients are pretty consistent between the two,
lending evidence to the idea that there is only a single equilibrium. Hence, it is globally concave.
=#



# Question 3 (Using the same size to try to understand how the problem works better)
    N = 5
    K = 5
    J = 5
    function logitsim(N, K, J)
        X = rand(N, K)                                              # Part A
        beta = [0.1 0.1 0.2 0.3 0; 0.5 0.8 0.13 0.22 0; 0.35 0.57 0.92 0.149 0; 0.241 0.390 0.631 0.1021 0; 0.1652 0.2673 0.4325 0.6998 0]      # Part B
        p = exp.(X*beta)./sum.(eachrow(X*beta))                      # Part C
        epsilon = rand(N)                                           # Part D
        y = zeros(N)                                                # Part E
        for i = 1:N
            if 0 <= epsilon[i] & epsilon[i] <= p[i, 1]
            y[i] = 1 
            elseif p[i, 1] < epsilon[i] & epsilon[i] <= p[i, 2]
            y[i] = 2            
            elseif p[i, 2] < epsilon[i] & epsilon[i] <= p[i, 3]
            y[i] = 3
            elseif p[i, 3] < epsilon[i] & epsilon[i] <= p[i, 4]
            y[i] = 4
            else
            y[i] = 5
            end
        end
        return(y)
    end    



# Question 4
    # Skip per Dr. Ransom's email



# Question 5        # No sigma per Dr. Ransom's Office Hours                Copy lines of code from 3c - 3e and put them in number 5; ytilde[i, d] in for loop
    D = 1
    function ols_smm(beta, X, y, D)
        K = size(X, 2)
        N = size(y, 1)
        if length(beta) == 1
            beta = beta[1]
        end
        p = exp.(X * beta) ./ sum.(eachrow(X * beta))               # Part C
        Random.seed!(1234)
        epsilon = rand(N)                                           # Part D
        ytilde = zeros(N)                                           # Part E
        for d in 1:D
            for i in 1:N
                if 0 <= epsilon[i] <= p[i, 1]
                ytilde[i, d] = 1 
                elseif p[i, 1] < epsilon[i] & epsilon[i] <= p[i, 2]
                ytilde[i, d] = 2            
                elseif p[i, 2] < epsilon[i] & epsilon[i] <= p[i, 3]
                ytilde[i, d] = 3
                elseif p[i, 3] < epsilon[i] & epsilon[i] <= p[i, 4]
                ytilde[i, d] = 4
                else
                ytilde[i, d] = 5
                end
            end
        end
        g = y .- mean(ytilde; dims = 2)
        J = g' * I * g
        return J
    end

    start_vals_smm = rand(size(X, 2))
    theta_hat_optim_smm = optimize(beta -> ols_smm(beta, X, y, D), start_vals_smm, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100000, show_trace = true, show_every = 50))
    println(theta_hat_optim_smm.minimizer)

#=
I am having quite a bit of trouble getting this to work. It seems that the error I am having is coming from the specification of D.
If I do not set it, I get an error saying that D is not defined. If it is in the function, an array issue is created at [1,2] on line 148.
The same is true if I set it above the value of 1.
=#

end

#ps7()