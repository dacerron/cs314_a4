# Load data
dataTable = readcsv("animals.csv")
X = float(real(dataTable[2:end,2:end]))
(n,d) = size(X)

# Standardize columns
include("misc.jl")
include("PCA.jl")
(X,mu,sigma) = standardizeCols(X)

model = myPCA(X,2)
Z = model.compress(X)

# Plot matrix as image
using PyPlot
figure(1)
clf()
imshow(X)

# Show scatterplot of 2 random features
figure(2)
clf()
plot(Z[:,1],Z[:,2],".")
for i in rand(1:n,10)
    annotate(dataTable[i+1,1],
	xy=[Z[i,1],Z[i,2]],
	xycoords="data")
end
