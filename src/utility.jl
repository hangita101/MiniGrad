function topo_graph(v::Node)
    topo = Vector{Node}()
    visited = Set{Node}()
    function build_topo(node::Node)
        if !(node in visited)
            push!(visited, node)

            for child in node._prev
                build_topo(child)
            end

            push!(topo, node)
        end
    end
    build_topo(v)
    return reverse(topo)
end

function backward!(val::Node)
    topo = topo_graph(val)
    val.grad = 1.0
    for node in topo
        if isempty(node._prev)
            continue
        end
        node._backward()
    end
end
