### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ eb3d7634-57af-4ab5-ba3f-9af11cd136af
using DelimitedFiles

# ╔═╡ 4dc416c4-ff16-4ae4-9edf-75a47dc926b0
using Statistics

# ╔═╡ 035de3c5-e99a-4791-8a89-51e9ccd3dffd
irisPath = "D:/Machine Learning/Datasheet/iris-train.txt"

# ╔═╡ 7f55931d-b8e1-4ad2-9d09-13d23533360a
A = readdlm(irisPath)

# ╔═╡ 55e870da-66e9-4634-8a3a-8376144a92a2
X_train = A[:,2:end]

# ╔═╡ 3e090e41-de58-4c7e-876c-949114d89867
Y_train = Int.(A[:,1])

# ╔═╡ aee5a161-63e8-48a4-9aa6-7795cd742cd7
function readData(path)
	A = readdlm(path)
	y = Int.(A[:,1])
	X = A[:,2:end]
	(X,y)
end

# ╔═╡ b82249a6-2e65-4d20-9294-d13f50be8a4f
X, y = readData(irisPath)

# ╔═╡ 4717acf7-7b17-487c-b3af-01c0e4d2a14a
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

# ╔═╡ b6323679-6bf3-4902-bc3c-aa180b4f917b
μ, σ, θ = train(X, y)

# ╔═╡ acab45ec-e1ec-476f-9be3-5f745d6a18f6
md"""
$\log P(y=k|x) = \sum_{j=1}^{D} \log P(x_j|y=k) + \log P(y=k)$
$=\sum_{j=1}^{D} \left[ -\log (\sqrt{2\pi}) -\log (\sigma_{jk} - \frac{1}{2\sigma_{jk}^2}(x_j - \mu_{jk})^2 \right] + \log θ_k$
$\propto -\sum_{j=1}^{D} \left[\log (\sigma_{jk}) + \frac{1}{2\sigma_{jk}^2}(x_j - \mu_{jk})^2 \right] + \log θ_k$
"""

# ╔═╡ bd1ff1f9-2384-47b3-9d0d-387212bc8404
function classify(μ, σ, θ, x)
	D, K = size(μ)
	
	p = zeros(K) #log posterior distribution
	for k = 1:K
		p[k] = -sum(log.(σ[:,k]) + (x - μ[:,k]).^2 ./ (2*σ[:,k].^2)) + log(θ[k])
	end
	argmax(p)
end

# ╔═╡ b7a9a2e7-3520-4624-a886-91dba78d5e2f
classify(μ, σ, θ, X[end,:])

# ╔═╡ 1e856cd0-301c-4025-8120-bf6e17508211
ŷ = [classify(μ, σ, θ, X[i,:]) for i = 1:length(y)]

# ╔═╡ 8eefefad-9226-4119-aa16-ca022676e904
sum(ŷ .== y)

# ╔═╡ e446998f-ad27-465e-9d4c-169df757e240
training_accuracy = sum(ŷ .== y) / length(y)

# ╔═╡ a9755824-b929-41bd-a8e0-3a4f1d12372d
irisPathTest = "D:/Machine Learning/Datasheet/iris-test.txt"

# ╔═╡ 7fe94433-db7e-4083-815a-3283e7c54c7a
X_t, y_t = readData(irisPathTest)

# ╔═╡ 40de9dea-6e5e-4feb-bb45-f21e3162790b
ŷ1 = map(i -> classify(μ, σ, θ, X_t[i,:]), 1:length(y_t))

# ╔═╡ 77161b4f-fc8d-40dd-aef1-e206f940d299
sum(ŷ1 .== y_t) / length(y_t)

# ╔═╡ 2263ea54-f0ec-4d6b-9866-11bff877aff0


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
# ╠═eb3d7634-57af-4ab5-ba3f-9af11cd136af
# ╠═4dc416c4-ff16-4ae4-9edf-75a47dc926b0
# ╠═035de3c5-e99a-4791-8a89-51e9ccd3dffd
# ╠═7f55931d-b8e1-4ad2-9d09-13d23533360a
# ╠═55e870da-66e9-4634-8a3a-8376144a92a2
# ╠═3e090e41-de58-4c7e-876c-949114d89867
# ╠═aee5a161-63e8-48a4-9aa6-7795cd742cd7
# ╠═b82249a6-2e65-4d20-9294-d13f50be8a4f
# ╠═4717acf7-7b17-487c-b3af-01c0e4d2a14a
# ╠═b6323679-6bf3-4902-bc3c-aa180b4f917b
# ╠═acab45ec-e1ec-476f-9be3-5f745d6a18f6
# ╠═bd1ff1f9-2384-47b3-9d0d-387212bc8404
# ╠═b7a9a2e7-3520-4624-a886-91dba78d5e2f
# ╠═1e856cd0-301c-4025-8120-bf6e17508211
# ╠═8eefefad-9226-4119-aa16-ca022676e904
# ╠═e446998f-ad27-465e-9d4c-169df757e240
# ╠═a9755824-b929-41bd-a8e0-3a4f1d12372d
# ╠═7fe94433-db7e-4083-815a-3283e7c54c7a
# ╠═40de9dea-6e5e-4feb-bb45-f21e3162790b
# ╠═77161b4f-fc8d-40dd-aef1-e206f940d299
# ╠═2263ea54-f0ec-4d6b-9866-11bff877aff0
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
