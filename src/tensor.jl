struct Tensor
    size::Vector{Int}
    data::Union{Matrix{Node},Vector{Node}}
end


function tensor(data::Union{Matrix,Vector})
    Tensor(collect(size(data)), Node.(data))
end

function tensor(size)
    return Tensor(collect(size), Node.(rand(size)))
end
rand(2,3)