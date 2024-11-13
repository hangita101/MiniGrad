module MiniGrad


# Define the Node struct and associated functions
mutable struct Node <: Number
    data::Number
    grad::Number
    _prev::Set{Node}
    op::String
    _backward::Function

    # Define constructors for Node
    function Node(data::Number)
        return new(data, 0.0, Set{Node}(), " ")
    end

    function Node(data::Number, _children::Union{Tuple{Node},Tuple{Node,Node}}, ope::String)
        return new(data, 0.0, Set(_children), ope)
    end

end


include("operations.jl")
include("utility.jl")

# Now, export all the necessary functions and types
export +, *, -, exp, tanh, inv, show, Node, backward!

end # module MiniGrad
