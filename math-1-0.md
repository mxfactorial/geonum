theres no such thing as a directionless number

printing signed 1d numbers on human consumable bi reports is fine

but they turn into complexity grenades when passed as inputs to physical models and data pipelines

oversimplified abstractions like "the scalar" are introduced into math because anyone can extend a manual runtime language with whatever they want

so it ends up polluted with a bunch of complicated, idle formalism and jargon

#### cart before the horse

dimensions are not things you add **after defining** sets and operations

dimensions need to ship with the cost of their transformation attached

and they need to be programmed with that value **before learning** which sets and operations they support

so a scalable algebra depends not on defining numbers and operations first, but on a choice of dimension that enables defining how magnitude **and** angle change between all the others, or the *shape of change*

#### the geometric number

physics and compute friendly math depends on dimensions referencing at least 2 values, or a 2-tuple, if they are to *cooperate* with each other when *operating* on a number: `[magnitude, angle]`

requiring an explicit angle for numbers in a polar vector avoids a "negative number":

`[4, pi] = -4`

`[4, 0] + [4, pi] = [0, 0]`

and an "imaginary unit":

`[4, pi]^(1/2) = [2, pi/2]`

now "complex numbers" are just `pi/2` multiples instead of some device people reference with a random character to dig themselves out of a "negative number" hole:

```
[1, 0]
[1, pi/2]
[1, pi]
[1, 3pi/2]
```

and building a "vector space" supporting an unlimited number of dimensions is just a recursion that starts with duplicating the first `[magnitude, angle]`, *orthogonally adding* the duplicate vector to the first to provide "quadratic closure", then "extending" the space by *orthogonally adding* as many dimensions as you want

the phrase "orthogonally adding" or `+ pi/2` is important here because it reveals the [minimum](https://en.wikipedia.org/wiki/Geodesic) angular toll added dimensions must pay to cross the discrete continuity constructed by the [initial](https://en.wikipedia.org/wiki/Initial_and_terminal_objects) `[[1, 0], [1, pi/2]]` pair:

`∫1dθ from 0 to pi/2 = + pi/2`

#### angles add, mags multiply

the definite integral explicitly reveals the "angles add, mags multiply" rule

this rule not only frees everyone up from toiling in a "matrix":
```
[1  0] × [cos(pi/2)   -sin(pi/2)] ≡ [1, 0] × [1, pi/2] = [1, pi/2]
         [sin(pi/2)    cos(pi/2)]
```

it also guides them to simply add `pi/2` to polar vector angles across orthogonally added dimensions:

`ijk = [1, 0 + pi/2] x [1, pi/2 + pi/2] x [1, pi + pi/2] = [1, 3pi] = [1, pi]`

no need to [devour](https://claude.ai/share/c9d9fb27-b50b-4a6c-8084-eab053461b27) cpu and memory by brute forcing the value through a "rank-3 tensor"

much less [dance](https://claude.ai/share/6bd33b37-6e5a-4f6a-8a18-42dd7e5f5d50) around orthogonalitys measure with a square

angles are signed since tangents *and* normals parameterize change as reflections in orthogonally added dimensions

"orientation" as signed rotations between geometric numbers enables support for "bivectors" and "nilpotency":
```
let v = [r1, θ1], w = [r2, θ2]
v ∧ w = [r1 x r2 x sin(θ2 - θ1), (θ1 + θ2 + pi/2) mod 2pi]
v ∧ v = 0 when sin(0) = 0
```
"differentiation" is automatic here: `v' = [r, θ + pi/2] when sin(x)' = cos(x) = sin(x + pi/2)`

and `[4, 0] ⋅ [4, pi] + [4, 0] ∧ [4, pi] = [16, pi] + [0, 0]` isnt an [apples and oranges](https://math.stackexchange.com/questions/3193125/intuition-for-geometric-product-being-dot-wedge-product) problem anymore: `[magnitude, angle]` and `[area, orientation]`

#### capture the flag

math would be useless if it didnt answer to physics

but once the power of 1 `apple` + 1 `apple` = 2 `apple` was discovered

someone ran off with the equal sign and used it to say `let set = {}`

rejecting the assignment to a fictional data type captures the flag from "mathematicians"

and `let space = sin(n pi/2)` brings it back home to physics

so goodbye "negative scalars" and goodbye to all the complexity you gave us

and goodbye "linear" vs "geometric" algebra

dont forget to take "over a field" with you when you leave

math not disciplined by application is a mess and we're obligated to keep tolls to a minimum