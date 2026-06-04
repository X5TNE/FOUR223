local NN = dofile("../nn.lua")
local SCALE = 25
-- these exist because tanh sucks and I love GELU more, but I use tanh anyway 😭 (slowed down + reverb)
local function normalize(x)
    return x / SCALE
end
local function denormalize(x)
    return x * SCALE
end
local model = NN.load(
    "addition_model.four",
    2,-- input size
    6-- manipulation layer count
)
local function predict(a, b)
    local output = model:forward({
        normalize(a),
        normalize(b)
    })
    return denormalize(output)
end

local a = math.random(0,10)
local b = math.random(0,10)
local c = a + b
local d = math.abs(predict(a, b))

print(a .. "+" .. b .. "=" .. c .. " | prediction: ~" .. d)