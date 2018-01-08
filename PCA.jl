include("misc.jl")

function PCA(X,k)
    (n,d) = size(X)

    # Subtract mean
    mu = mean(X,1)
    X -= repmat(mu,n,1)

    (U,S,V) = svd(X)
    W = V[:,1:k]'

    compress(Xhat) = compressFunc(Xhat,W,mu)
    expand(Z) = expandFunc(Z,W,mu)

    return CompressModel(compress,expand,W)
end

function myPCA(X,k)
    (n,d) = size(X)
    
    mu = mean(X,1)
    X -= repmat(mu,n,1)
    
    (W,Z) initPCA(X,k)
    
    W = solveW(W,Z,X)
    Z = solveZ(W,Z,X)
    
    compress(Xhat) = compressFunc(Xhat,W,mu)
    expand(Z) = expandFunc(Z,W,mu)

    return CompressModel(compress,expand,W) 
end

function compressFunc(Xhat,W,mu)
    (t,d) = size(Xhat)
    Xcentered = Xhat - repmat(mu,t,1)
    return Xcentered*W' # Assumes W has orthogonal rows
end

function expandFunc(Z,W,mu)
    (t,k) = size(Z)
    return Z*W + repmat(mu,t,1)
end

function solveW(W,Z,X)
    return (Z'Z)\(Z'X)
end

function solveZ(W,Z,X)
    return XW'\(WW')
end

function initPCA(X,k)
    (n,d) = size(X)
    W = rand(k,d)
    Z = rand(n,k)

    return (W,Z) 
end
