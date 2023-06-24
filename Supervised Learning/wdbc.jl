### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ b5b2822a-112f-4dba-972a-cc60fe46f97f
using DelimitedFiles

# ╔═╡ 0afb22d4-a9b8-46b2-8f9c-a6b0e429c9b7
using Statistics

# ╔═╡ baea5b6a-944a-4e88-8f0a-ed349640d530
wdbcPath = "D:/Machine Learning/Datasheet/wdbc.txt"

# ╔═╡ 3b55204b-f3a4-4a1f-828f-2683d18d32c7
A = readdlm(wdbcPath,',')

# ╔═╡ 170a25a6-bde0-4913-80bd-10ea2fd9bcdc
X = A[:,3:end]

# ╔═╡ 3e0356a0-d0c5-4b22-908f-0ceab02a9449
y = Int.(A[:,2]) .+ 1

# ╔═╡ 53b7dd49-267c-4755-bab9-f3ec79270997
function train(X, y)
	K = length(unique(y))
	N, D = size(X)
	μ = zeros(D,K)
	σ = zeros(D,K)
	θ = zeros(K) #prior θ[k] = P(y=k)
	for k = 1:K
		idk = (y .== k)
		Xk = X[idk,:]
		μ[:,k] = mean(Xk, dims = 1)
		σ[:,k] = std(Xk, dims = 1)
		θ[k] = sum(idk) / N
	end
	(μ, σ, θ)
end

# ╔═╡ 98021b20-6fc9-4b7e-b362-461efa57b233
μ, σ, θ = train(X,y)

# ╔═╡ 25930359-c6c2-4f58-87ed-4eb6c712dda6
y .+ 1

# ╔═╡ 06294e4b-1968-4939-933f-887987507dc4


# ╔═╡ a42c58c9-018f-48d4-a55e-7e16ad2fb9c1
function classify(μ, σ, θ, x)
	D, K = size(μ)
	p = zeros(K)
	for k = 1:K
		p[k] = -sum(log.(σ[:,k]) + (x - μ[:,k]).^2 ./ (2*σ[:,k].^2)) + log(θ[k])
	end
	argmax(p)
end

# ╔═╡ 0eba9c65-9d4f-4861-ae09-1a5c89f59278
z = map(i -> classify(μ, σ, θ, X[565,:]), 1:length(y))

# ╔═╡ 0cd81fbc-3f27-47d2-bbd8-262192501c24
classify(μ, σ, θ, X[563,:])

# ╔═╡ ea312410-5d67-411e-bc5e-d4841831cb22
ŷ = map(i -> classify(μ, σ, θ, X[i,:]), 1:length(y))

# ╔═╡ 356ddfed-1319-4845-86a2-902224666190
training_accuracy = sum(ŷ .== y) / length(y)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "f8aed8cc7ec98e25caba5c40ea614d484439ba58"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"
"""

# ╔═╡ Cell order:
# ╠═b5b2822a-112f-4dba-972a-cc60fe46f97f
# ╠═0afb22d4-a9b8-46b2-8f9c-a6b0e429c9b7
# ╠═baea5b6a-944a-4e88-8f0a-ed349640d530
# ╠═3b55204b-f3a4-4a1f-828f-2683d18d32c7
# ╠═170a25a6-bde0-4913-80bd-10ea2fd9bcdc
# ╠═3e0356a0-d0c5-4b22-908f-0ceab02a9449
# ╠═53b7dd49-267c-4755-bab9-f3ec79270997
# ╠═98021b20-6fc9-4b7e-b362-461efa57b233
# ╠═25930359-c6c2-4f58-87ed-4eb6c712dda6
# ╠═06294e4b-1968-4939-933f-887987507dc4
# ╠═a42c58c9-018f-48d4-a55e-7e16ad2fb9c1
# ╠═0eba9c65-9d4f-4861-ae09-1a5c89f59278
# ╠═0cd81fbc-3f27-47d2-bbd8-262192501c24
# ╠═ea312410-5d67-411e-bc5e-d4841831cb22
# ╠═356ddfed-1319-4845-86a2-902224666190
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
