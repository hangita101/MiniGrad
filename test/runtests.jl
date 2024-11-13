using MiniGrad
using Test
using Random

x = randn()
y = randn()
temp1 = Node(x)
temp2 = Node(y)

function Tst(f, x, y)
    @test f(x) == f(y).data
end

function sigmoid(x)
    return 1.0 / (1.0 + exp(-x))
end

function relu(x)
    return max(0, x)
end

@testset "Forward Pass test" begin
    @test x + y == (temp1 + temp2).data
    @test x * y == (temp1 * temp2).data
    @test -x == MiniGrad.negate(temp1).data
    @test x - y == (temp1 - temp2).data
    @test temp1 == temp1
    @test exp(temp1).data == exp(x)
    Tst(tanh, x, temp1)
    Tst(inv, x, temp1)
    @test isapprox(1 / x, (1 / temp1).data, atol=0.001)
    @test isapprox(x / y, (temp1 / temp2).data, atol=0.001)
    @test sigmoid(x) == sigmoid(temp1.data)
    @test relu(x) == MiniGrad.relu(temp1).data
    @test isless(temp1,-Inf)==false
end

