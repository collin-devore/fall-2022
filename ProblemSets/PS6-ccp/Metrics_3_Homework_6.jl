# Problem Set 3
    # Collin DeVore
        # with Bowei Dong

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV
include("create_grids.jl")

function ps6()
# Question 1
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    df = @transform(df, :bus_id = 1:size(df, 1))
    vary1 = @select(df, :bus_id, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10, :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20, :RouteUsage, :Branded)
    vary2 = DataFrames.stack(vary1, Not([:bus_id, :RouteUsage, :Branded]))
    rename!(vary2, :value => :Y)
    vary2 = @transform(vary2, :time = kron(collect([1:20]...), ones(size(df, 1))))
    select!(vary2, Not(:variable))

    x1 = @select(df, :bus_id, :Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, :Odo15, :Odo16, :Odo17, :Odo18, :Odo19, :Odo20)
    x2 = DataFrames.stack(x1, Not([:bus_id]))
    rename!(x2, :value => :Odometer)
    x2 = @transform(x2, :time = kron(collect([1:20]...), ones(size(df, 1))))
    select!(x2, Not(:variable))

    df_long = leftjoin(vary2, x2, on = [:bus_id, :time])
    sort!(df_long, [:bus_id, :time])



# Question 2 (Inspired by https://www.machinelearningplus.com/julia/logistic-regression-in-julia-practical-guide-with-examples/)
    form1 = @formula(Y ~ Odometer * Odometer * RouteUsage * RouteUsage * Branded * time * time)
    logmod1 = glm(form1, df_long, Binomial(), ProbitLink())
    println(logmod1)



# Question 3 (With a ton of help from the PS5 solutions and Dr. Ransom)
    # Part A
    zval, zbin, xval, xbin, xtran = create_grids()


    # Part B
    df2 = DataFrame(Odometer = kron(ones(zbin), xval), RouteUsage = kron(ones(xbin), zval), Brand = zeros(20301), time = zeros(20301))

    function opstopccp(df2, logmod1, Xst, Zst, xtran, Y, theta)
        beta = 0.9
        FV = zeros(length(xtran, 1), 2, T+1)
        for t in 2:T
            for i in 0:1
                df2.time .= t
                df2.Branded .= b
                p0 = 1 - predict(logmod1, df2)
                FV[length(xtran, 1), b + 1, t] = -beta.*log(p0)         # Stores theta values and log likelihood?
            end
        end
        for s = 1:N
            for t = 1:T
                row0 = (d.Zst[s] - 1) * d.xbin + 1
                row1 = d.Xst[s, t] + (d.Zst[s] - 1) * xbin
                EFVT1[s, t] = (xtran[row1, :].-xtran[row0, :])'*FV[row0:row0 + xbin - 1, B[s] + 1, t+1]
                return EFVT1'[:]
            end
        end
    end


    # Part C (From the Problem Set)
    df_long = @transform(df_long, :FV = EFVT1)
    theta_hat_ccp_glm = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink(), offset = df_long.FV)
    println(theta_hat_ccp_glm)
    


    # Part D
    logmod2 = glm(opstopccp, df_long, Binomial(), ProbitLink(), offset = EFVT1)
    println(logmod2)
end


    # Part E
ps6()

    # Part F
# Glory to the power of the CCPs!!!!!
