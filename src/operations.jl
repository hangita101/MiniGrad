import Base: show
import Base: +
import Base: *
import Base: -
import Base: ==
import Base: hash
import Base: exp
import Base: tanh
import Base: inv
import Base: ^
import Base:negate

"This thing is hack. Its promoting any type lower then Node to Node type."
Base.promote_rule(::Type{<:Node}, ::Type{T}) where {T<:Number} = Node

function show(io::IO, node::Node)
    println(io, "Value: ", node.data)
end

function +(a::Node, b::Node)

    out = Node(a.data + b.data, (a, b), "+")
    function backward()
        a.grad += 1.0 * out.grad
        b.grad += 1.0 * out.grad
    end
    out._backward = backward
    return out
end

function *(a::Node, b::Node)
    out = Node(a.data * b.data, (a, b), "*")
    function backward()
        a.grad += b.data * out.grad
        b.grad += a.data * out.grad
    end
    out._backward = backward
    return out
end

function negate(a::Node)
    return a*-1
end


function -(a::Node, b::Node)
    out = Node(a.data - b.data, (a, b), "*")
    function backward()
        a.grad += 1.0 * out.grad
        b.grad += -1.0 * out.grad
    end
    out._backward = backward
    return out
end


function ==(a::Node, b::Node)
    return a === b
end

function hash(a::Node)
    return hash(objectid(a))
end

function exp(a::Node)
    out = Node(exp(a.data), (a,), "exp")

    function backward()
        a.grad += exp(a.data) * out.grad
    end
    out._backward = backward
    return out
end

function tanh(a::Node)
    out = Node(tanh(a.data), (a,), "tanh")
    function backward()
        a.grad += (1 - tanh(a.data)^2) * out.grad
    end

    out._backward = backward

    return out
end

function inv(a::Node)
    out = Node(inv(a.data), (a,), "inv")

    function backward()
        a.grad += out.grad * -1 * (a.data)^2
    end

    out._backward = backward
    return out

end

function ^(a::Node, b::Real)
    out = Node(a.data^b, (a,), "^")

    function backward()
        a.grad += (b * a.data^(b - 1)) * out.grad
    end
    a._backward = backward
    return out
end

function relu(a::Node)
    out = Node(a.data < 0 ? 0 : a.data, (a,), "RelU")

    function backward()
        a.grad += (out.data > 0) * out.grad
    end

    out._backward = backward

end

function sigmoid(a::Node)
    return 1/1+exp(negate(a))
end