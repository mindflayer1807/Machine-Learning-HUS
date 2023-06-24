### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 12924880-c7b8-11ed-32a1-0fd1281d737e
using DelimitedFiles

# ╔═╡ 769c1eb7-84fd-460e-9a6e-9245da8ebb98
function readData(path)
	A = readdlm(path, ',')
	y = A[:,2]
	X = A[:,1]
	X, y
end

# ╔═╡ 5726c855-4727-4e99-acc8-6367f49251e6
path = "D:/Machine Learning/Datasheet/profits.txt"

# ╔═╡ 8cd6e2b3-f487-4596-803e-9db6b81da685
X , y = readData(path)

# ╔═╡ 358fba9b-400a-4ba7-a508-ff5cc0685c9d


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "739f2f3af706c750957bd6ec39e0874ba8eb265d"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
"""

# ╔═╡ Cell order:
# ╠═12924880-c7b8-11ed-32a1-0fd1281d737e
# ╠═769c1eb7-84fd-460e-9a6e-9245da8ebb98
# ╠═5726c855-4727-4e99-acc8-6367f49251e6
# ╠═8cd6e2b3-f487-4596-803e-9db6b81da685
# ╠═358fba9b-400a-4ba7-a508-ff5cc0685c9d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
