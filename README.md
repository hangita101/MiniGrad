# An Mini Reverse Mode AD (Backpropagation) in Julia

How to use?

1. Clone the repo
2. Start a Julia REPL
3. `using Pkg` then `Pkg.add(path to that repo)` (start the Julia REPL outside the cloned Repo)
4. Import it using `using MiniGrad`

### Example:
```julia
julia> x= Node(2.0)
2.0

julia> y=x^3+5x
18.0

julia> backward!(y)

julia> y.grad
1.0

julia> x.grad
17.0

```
It just calcualtes the the `dY_dx` of the equation

Each node has an `grad` attrubite

We calculate it by applying `backward!` function on the result of the expression containing `Node` struct.



There are still many thing remainig to be added.

<hr>
This project is based on :

[Andrej Karpathy intro to Neural Network](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
