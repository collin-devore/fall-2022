# Problem Set 1
   # Collin DeVore
      # With Bowei Dong and William Myers


# Question 1 
   # A
using JLD2, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions
function q1()
   Random.seed!(1234)
   A = rand(Uniform(-5,10), (10,7))
   B = rand(Normal(-5,15), (10,7))
   C = hcat(A[1:5,1:5],B[1:5,6:7])
   x3 = collect(A.<0)
   D = A.*x3

   # B
   length(A)

   # C
   length(unique(D))

   # D
   E = reshape(B,(70,1))
   E = vec(B)              # This is an easier way to get the same vector

   # E
   F = [A;;;B]

   # F
   x1 = permutedims(F,[3,1,2])
   F = x1

   # G
   G = kron(B,C)
   kron(C,F)      # This gives an error since the matrices are not the same size

   # H
   jldsave(raw"C:\\Users\\Colli\\Documents\\002 - OU Doctoral Economics\\Year 2\\Semester 1\\Metrics 3\\matrixpractice.jld2"; A, B, C, D, E, F, G)

   # I
   jldsave(raw"C:\\Users\\Colli\\Documents\\002 - OU Doctoral Economics\\Year 2\\Semester 1\\Metrics 3\\firstmatrix.jld2"; A, B, C, D)

   # J
   C1 = DataFrame(C, :auto)
   CSV.write(raw"C:\\Users\\Colli\\Documents\\002 - OU Doctoral Economics\\Year 2\\Semester 1\\Metrics 3\\Cmatrix.csv", C1)

   # K
   D1 = DataFrame(D, :auto)
   CSV.write(raw"C:\\Users\\Colli\\Documents\\002 - OU Doctoral Economics\\Year 2\\Semester 1\\Metrics 3\\Dmatrix.dat", D1)
   return A,B,C,D
end
   A,B,C,D = q1()


# Question 2
   # A
function q2(A,B,C)
   Random.seed!(1234)
   A = rand(Uniform(-5,10), (10,7))
   B = rand(Normal(-5,15), (10,7))

      # With a Loop
   AB = A
   for i in 1:size(A,1)
      for j in 1:size(A,2)
         AB[i,j] = A[i,j] * B[i,j]
      end
   end
   print(AB)

      # Without a Loop
   AB2 =.*(A,B)


   # B
      # With a Loop
   Random.seed!(1234)
   C = hcat(A[1:5,1:5],B[1:5,6:7])

   x4 = [(5 >= x >= -5) for x in C]
   vec(x4)
   Cnewvec = x4.*C

   for n in Cnewvec
      Cprime = filter!(n->n!=0, Cnewvec)
   end

      # Without a Loop
   Cprime2 = filter(x -> -5 ≤ x ≤ 5,transpose(vec(C)))


   # C
   # Method 1
   Col1 = ones(15169,1)

   Col5 = rand(Binomial(20,0.6), (15169,1))

   Col6 = rand(Binomial(20,0.5), (15169,1))

   for t in 1:5
      Col2t = rand(Uniform(0,1),(15169,1))
      for i in 1:15169
         if Col21[i,1] <= ((0.75*(6-t))/5)
            Col21[i,1] = 1
         else
            Col21[i,1] = 0
         end
      end

      Col3t = rand(Normal(15+t-1, 5*(t-1)),(15169,1))

      Col4t = rand(Normal(π*(6-t)/3, 1/ℯ),(15169,1))
   end

   # Method 2 (Same answer)
   t=1
   Col1 = ones(15169,1)

   Col21 = rand(Uniform(0,1),(15169,1))
   for i in 1:15169
      if Col21[i,1] <= ((0.75*(6-t))/5)
         Col21[i,1] = 1
      else
         Col21[i,1] = 0
      end
   end

   Col31 = rand(Normal(15+t-1, 5*(t-1)),(15169,1))

   Col41 = rand(Normal(π*(6-t)/3, 1/ℯ),(15169,1))

   Col5 = rand(Binomial(20,0.6), (15169,1))

   Col6 = rand(Binomial(20,0.5), (15169,1))


   t = 2

   Col22 = rand(Uniform(0,1),(15169,1))
   for i in 1:15169
      if Col21[i,1] <= ((0.75*(6-t))/5)
         Col21[i,1] = 1
      else
         Col21[i,1] = 0
      end
   end

   Col32 = rand(Normal(15+t-1, 5*(t-1)),(15169,1))

   Col42 = rand(Normal(π*(6-t)/3, 1/ℯ),(15169,1))


   t = 3

   Col23 = rand(Uniform(0,1),(15169,1))
   for i in 1:15169
      if Col21[i,1] <= ((0.75*(6-t))/5)
         Col21[i,1] = 1
      else
         Col21[i,1] = 0
      end
   end

   Col33 = rand(Normal(15+t-1, 5*(t-1)),(15169,1))

   Col43 = rand(Normal(π*(6-t)/3, 1/ℯ),(15169,1))


   t = 4

   Col24 = rand(Uniform(0,1),(15169,1))
   for i in 1:15169
      if Col21[i,1] <= ((0.75*(6-t))/5)
         Col21[i,1] = 1
      else
         Col21[i,1] = 0
      end
   end

   Col34 = rand(Normal(15+t-1, 5*(t-1)),(15169,1))

   Col44 = rand(Normal(π*(6-t)/3, 1/ℯ),(15169,1))


   t = 5

   Col25 = rand(Uniform(0,1),(15169,1))
   for i in 1:15169
      if Col21[i,1] <= ((0.75*(6-t))/5)
         Col21[i,1] = 1
      else
         Col21[i,1] = 0
      end
   end

   Col35 = rand(Normal(15+t-1, 5*(t-1)),(15169,1))

   Col45 = rand(Normal(π*(6-t)/3, 1/ℯ),(15169,1))



   X = [Col1;;Col21;;Col31;;Col41;;Col5;;Col6;;;Col1;;Col22;;Col32;;Col42;;Col5;;Col6;;;Col1;;Col23;;Col33;;Col43;;Col5;;Col6;;;Col1;;Col24;;Col34;;Col44;;Col5;;Col6;;;Col1;;Col25;;Col35;;Col45;;Col5;;Col6]


   # D
T = 5
y1 = [0.75 + (0.25t) for t in 1:T]
ColD1 = reshape(y1,(1,T))

y2 = [log(t) for t in 1:T]
ColD2 = reshape(y2,(1,T))

y3 = [- sqrt(t) for t in 1:T]
ColD3 = reshape(y3,(1,T))

y4 = [(ℯ^t)-(ℯ^(t+1)) for t in 1:T]
ColD4 = reshape(y4,(1,T))

y5 = [t for t in 1:T]
ColD5 = reshape(y5,(1,T))

y6 = [t/3 for t in 1:T]
ColD6 = reshape(y6,(1,T))

β = cat(ColD1,ColD2,ColD3,ColD4,ColD5,ColD6,dims=1)

print(β)


   # E

   ε = rand(Normal(0,0.36), (5,1))

   Y = X*β + ε
end

q2(A,B,C)



# Question 3
   # A
using JLD2, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions
nlsw88 = CSV.read(raw"C:\\Users\\Colli\\Documents\\002 - OU Doctoral Economics\\Year 2\\Semester 1\\Metrics 3\\nlsw88.csv", DataFrame)
jldsave("nlsw88.jld2";nlsw88)

function q3()
   # B
freqtable(nlsw88.never_married)
234/(2012+234)       # 0.10418521816562779, so about 10.4% have never been married.
freqtable(nlsw88.collgrad)
532/(1714+532)       # 0.23686553873552982, so about 23.7% have never been married.

   # C
freqtable(nlsw88.race)
      # for race 1
1637/(1637+583+26)   # 0.7288512911843277, so about 72.9% are in race cagory 1.
      # for race 2
583/(1637+583+26)    # 0.25957257346393586, so about 26.0% are in race category 2.
      # for race 3
26/(1637+583+26)     # 0.01157613535173642, so about 1.2% are in race category 3.


   # D
summarystats = describe(nlsw88)
describe(nlsw88.grade)     # 2 observations are missing

   # E
freqtable(nlsw88.industry, nlsw88.occupation)

   # F
wagetable = select(nlsw88,11,12,14)
end

q3()


# Question 4
   # A
firstmatrix = load("firstmatrix.jld2")
A = load("firstmatrix.jld2","A")
B = load("firstmatrix.jld2","B")
C = load("firstmatrix.jld2","C")
D = load("firstmatrix.jld2","D")
nlsw88 = CSV.read("C:\\Users\\Colli\\Documents\\002 - OU Doctoral Economics\\Year 2\\Semester 1\\Metrics 3\\firstmatrix.jld2", DataFrame)
function q4()
   # B
   function matrixops(A1,B1)
      if size(A1) != size(B1)
         return "inputs must have the same size"
      else
         return A1.*B1, A1'*B1, A1+B1   # This function returns the inputs for element-by-element multiplication, transpose element-by-element multiplication, and the addition of the two matrices.
      end
   end

   # C

   # D
   matrixops(A,B)

   # E

   # F
   matrixops(C,D)       # The function returns "inputs must have the same size" since the inputs were ran as matrices of different sizes.

   # G
   convert(Array,nlsw88.ttl_exp)
   convert(Array,nlsw88.wage)
   matrixops(nlsw88.ttl_exp,nlsw88.wage)
   end

q4()