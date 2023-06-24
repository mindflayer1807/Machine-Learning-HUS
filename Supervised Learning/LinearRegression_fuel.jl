### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 30575981-541a-4a55-b4fd-0fbdd86ed5d6
using DelimitedFiles

# ╔═╡ 3d328515-6118-4333-982d-2cf13acbb68c
using Statistics

# ╔═╡ 1b3cdf08-581c-462a-965d-f9e3a59ab21b
using LinearAlgebra

# ╔═╡ 67cc893f-8037-4201-afb6-b5a194bb8cc7
A = readdlm("D:/Machine Learning/Datasheet/fuel.txt",',')

# ╔═╡ eed9dee1-ccc7-4fa1-ab0f-5fc4a8f7c950
Tax = A[:, end]

# ╔═╡ 505599ad-ba12-4b87-997f-9ba811c42315
Dlic = 1000 .* (A[:,2]./A[:,7])

# ╔═╡ f32350ee-4efc-4e07-969a-a68fbf12b0a8
Income = A[:,4]

# ╔═╡ 54e3a30b-1dcf-4654-adf5-5f629115d737
logMiles = log2.(A[:,5])

# ╔═╡ bce7684e-646d-4b58-8bc4-6a28f5422813
N = length(Tax)

# ╔═╡ 8be8cc9a-fd6c-4b9e-8fbe-5bf44c007dc5
X = float.([ones(N) Tax Dlic Income logMiles])

# ╔═╡ 76b67bbc-f09c-4387-bcae-71be2eb22e6c
y = A[:, 3] ./ A[:,7] .* 1000

# ╔═╡ aeedd5c2-eda7-4019-a3d5-5f147b30b9bb
train(X,y) =  inv(X'*X)*X'*y

# ╔═╡ 296b64c7-ec67-4036-b2f3-61823bebea1b
θ = train(X,y)

# ╔═╡ 0a2e57e7-7949-402a-a344-0b6760e5305d
predict(θ, xNew) = xNew'*θ

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "b56cb346389a0059968eb60a6685acf8e6f54b3f"

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
# ╠═30575981-541a-4a55-b4fd-0fbdd86ed5d6
# ╠═67cc893f-8037-4201-afb6-b5a194bb8cc7
# ╠═eed9dee1-ccc7-4fa1-ab0f-5fc4a8f7c950
# ╠═505599ad-ba12-4b87-997f-9ba811c42315
# ╠═f32350ee-4efc-4e07-969a-a68fbf12b0a8
# ╠═54e3a30b-1dcf-4654-adf5-5f629115d737
# ╠═bce7684e-646d-4b58-8bc4-6a28f5422813
# ╠═8be8cc9a-fd6c-4b9e-8fbe-5bf44c007dc5
# ╠═76b67bbc-f09c-4387-bcae-71be2eb22e6c
# ╠═aeedd5c2-eda7-4019-a3d5-5f147b30b9bb
# ╠═296b64c7-ec67-4036-b2f3-61823bebea1b
# ╠═0a2e57e7-7949-402a-a344-0b6760e5305d
# ╠═3d328515-6118-4333-982d-2cf13acbb68c
# ╠═1b3cdf08-581c-462a-965d-f9e3a59ab21b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
