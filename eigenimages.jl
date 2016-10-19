"""
    labels = classify_image(tst, trn, k)

Input:
* `tst`: a n x T vector corresponding to T images that are to be classified
* `trn`: n x m x 10  matrix containing m training samples for each digit
* `k`: integer between 1 and n
"""
function classify_image(
    tst::Array{Float64,2},
    trn::Array{Float64,3},
    k::Int64
    )
    n, T = size(tst)
    m = size(trn, 2)
    # compute projection matrices:
    P = zeros(n, n, 10)
    for i in 1:10
        U, S, V = svd(trn[:,:,i])
        U1 = U[:,1:k]
        P[:,:,i] = U1*U1'
    end

    # find errors:
    sqerr = zeros(10, T)
    for i in 1:10
        sqerr[i,:] = sumabs2(tst - P[:,:,i]*tst, 1)
    end

    return [indmin(sqerr[:,i]) for i in 1:size(sqerr,2)] - 1
end

function linear_combo(a1,a2,a3,digit,trn)
    U,S,V = svd(trn[:,:,digit+1])
    U1, U2, U3 = U[:,1], U[:,2], U[:,3]
    v = a1*U1 + a2*U2 + a3*U3
    return v
end

function vec2mat(v)
    # shift to zero:
    v = (v - minimum(v))
    # scale to 1:
    v /= maximum(v)
    # shift to [-0.5,0.5]:
    v -= 0.5
    return reshape(v, 16, 16)'
end
