# Problem Set 3
    # Collin DeVore
        # with Bowei Dong

using DataFramesMeta, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV

function ps5()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

# Question 1
    # From Dr. Ransom's Github Problem Set files
    include("create_grids.jl")

    df = @transform(df, bus_id = 1:size(df, 1))
    vary1 = @select(df, :bus_id, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10, :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20, :RouteUsage, :Branded)
    vary2 = DataFrames.stack(vary1, Not([:bus_id, :RouteUsage, :Branded]))
    rename!(vary2, :value => :Y)
    vary2 = @transform(vary2, time = kron(collect([1:20]...), ones(size(df, 1))))
    select!(vary2, Not(:variable))

    x1 = @select(df, :bus_id, :Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, :Odo15, :Odo16, :Odo17, :Odo18, :Odo19, :Odo20)
    x2 = DataFrames.stack(x1, Not([:bus_id]))
    rename!(x2, :value => :Odometer)
    x2 = @transform(x2, time = kron(collect([1:20]...), ones(size(df, 1))))
    select!(x2, Not(:variable))

    df_long = leftjoin(y2, x2, on = [:bus_id, :time])
    sort!(df_long, [:bus_id, :time])



# Question 2 (Inspired by https://www.machinelearningplus.com/julia/logistic-regression-in-julia-practical-guide-with-examples/)

    form1 = @formula(Y ~ Odometer + Branded)
    log1 = glm(form1, df_long, Binomial(), ProbitLink())
    println(log1)
    # theta0 = 1.13352
    # theta1 = -0.083546
    # theta2 = 0.582591

#=
function logit(alpha, X, y)
    y2 = y
    x2 = X[2, 20000]
    b = X[2, end]
    P = exp.(X * alpha)./(1 .+ exp.(X * alpha))
    loglike = -sum(((y .== 1) .* log.(P)) .+ ((y .== 0) .* log.(1 .- P)))
    return loglike
end

alphahatoptim = optimize(a -> logit(a, X, y), rand(size(X, 2)), LBFGS(), Optim.Options(g_tol = 1e-6, iteration = 100000, show_trace = true))
println(alphahatoptim.minimizer)
=#



# Question 3
    # Part A (with some help from Dr. Ransom's PS5starter.jl file)
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    Y = Matrix(df[:, [:Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10, :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20]])
    Xst = Matrix(df[:, [:Xst1, :Xst2, :Xst3, :Xst4, :Xst5, :Xst6, :Xst7, :Xst8, :Xst9, :Xst10, :Xst11, :Xst12, :Xst13, :Xst14, :Xst15, :Xst16, :Xst17, :Xst18, :Xst19, :Xst20]])
    Zst = Matrix(df[:, [:Zst]])
    Odom = Matrix(df[:, [:Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, :Odo15, :Odo16, :Odo17, :Odo18, :Odo19, :Odo20]])



    # Part B (with some help from Dr. Ransom's PS5starter.jl file)
    zval, zbin, xval, xbin, xtran = create_grids()



#=
    # Part C
    @views @inbounds function opstop()                             # Part E and F
        FV = zeros(length(xtran, 1), 2, T + 1, dims = 3)
        for t in -T:1                                                                                                         # With help from https://discourse.julialang.org/t/iteration-reverse-in-for-loop/44551
            for i in 0:1
                for j in 1:length(zbin, 1)
                    for v in 1:length(xbin, 1)
                        x[v] + (z[j]-1)*xbin[v] 
                        v1 = xtran[row, :]'* FV[(z[j] - 1) * xbin[v] + 1: z[j] * xbin[v], i + 1, t + 1]
                        v0 = xtran[1 + (z[j] - 1) * xbin[v], :]'* FV[(z[j] - 1) * xbin[v] + 1: z[j] * xbin[v], i + 1, t + 1]
                        beta = 0.9
                        FV[t, :, 3] = beta .* log.(exp(v0) .+ exp(v1))

    # Part D (Switch this to move after the other loops, only loop over buses and time periods)
                        for bu in 1:length(bus_id, 1)
                            loglik = -sum(0)
                            stmrep = 1 + (:Zst[j] - 1) * xbin[v]
                            flucom = beta .* log.(exp(xtran[1 + (:Zst[j] - 1) * xbin[v], :]'* FV[(:Zst[j] - 1) * xbin[v] + 1: :Zst[j] * xbin[v], i + 1, t + 1]) .+ exp(xtran[row, :]'* FV[(:Zst[j] - 1) * xbin[v] + 1: :Zst[j] * xbin[v], i + 1, t + 1]))
                            (xtran[row1, :] .-xtran[row0, :])'* fva[row0: row0 + xbin - 1, B[i] + 1, t + 1]
                            P1 = (exp.(flucom))./(1 .+ exp.(flucom))
                            P0 = 1 .- P1
                            loglik = -sum(:Y .* log.(P[i]))
                            return loglik
                        end
                    end
                end
            end
        end
    end
=#



    # Part C
    @views @inbounds function opstop(theta, Xst, Zst, Y)                             # Part E and F
        FV = zeros(length(xtran, 1), 2, T + 1, dims = 3)
        for t in T:-1:1                                                                                                         # With help from https://discourse.julialang.org/t/iteration-reverse-in-for-loop/44551
            for i in 0:1
                for z in 1:zbin
                    for x in 1:xbin
                        row = x + (z-1)*xbin 
                        u1 = theta[1] .+ (theta[2] .* Xst) .+ (theta[3] .* b[i])
                        intv1 = xtran[row, :]'* FV[(z - 1) * xbin + 1: z * xbin, i + 1, t + 1]
                        v1 = u1 .+ (beta .* intv1)
                        u0 = 0
                        intv0 = xtran[1 + (z - 1) * xbin, :]'* FV[(z - 1) * xbin + 1: z * xbin, i + 1, t + 1]
                        v0 = intv0
                        beta = 0.9
                        FV[t, :, 3] = beta .* log.(exp(v0) .+ exp(v1))
                    end
                end
            end
        end

    # Part D
    
        for i in 1:length(bus_id, 1)
            for t = 1:length(Xst)
                loglik = -sum(0)
                row = 1 + (:Zst - 1) * xbin
                U1d = theta[1] .+ (theta[2] .* Xst) .+ (theta[3] .* B[i])
                IntV1d = beta .* log.(exp(xtran[1 + (:Zst - 1) * xbin, :]'* FV[(:Zst - 1) * xbin + 1: :Zst * xbin, i + 1, t + 1]) .+ exp(xtran[row, :]'* FV[(:Zst - 1) * xbin + 1: :Zst * xbin, i + 1, t + 1]))
                V1d = U1d .+ IntV1d
                U0d = 0
                IntV0d = (xtran[row1, :] .-xtran[row0, :])'* FV[row0: row0 + xbin - 1, B[i] + 1, t + 1]
                V0d = IntV0d
                P1 = (exp.(IntV1d .- IntV0d))./(1 .+ exp.(IntV1d .- IntV0d))
                P0 = 1 .- P1
                loglik = -(((Y[i, t] == 1) .* log.(P1)) .- ((Y[i, t] == 0) .* log.(P0)))
                return loglik
            end
        end
    end

    theta_zero_one = zeros(2 * length(xtran, 1))
    theta_zero = zeros(3 * size(Xst, 2) + 1)
    theta_hat_optim = optimize(theta -> opstop(theta, Xst, Zst, Y), theta_zero_one, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100000, show_trace = true, show_every = 50))
    theta_hat_mle = theta_hat_optim.minimizer
    println(theta_hat_mle)
end         # Part G



    # Part H
ps5()



    # Part I
# Done.




