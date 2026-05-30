-- train_add.lua
-- basically we train a FOUR model that adds numbers together even though FOUR could do better with binary compression and stuff.
math.randomseed(os.time())
local NN = dofile("nn.lua")
local model = NN.random(2, 6)
local SCALE = 5000
-- these exist because tanh sucks and I love GELU more, but I use tanh anyway 😭
function normalize(x)
    return x / SCALE
end
function denormalize(x)
    return x * SCALE
end
function generatesample()
    local a = math.random(0, 10)
    local b = math.random(0, 10)
    local result = a + b
    return {
        input = {
            normalize(a),
            normalize(b)
        },

        output = normalize(result)
    }
end
evaldata = {}
for x=1, 4444 do
    evaldata[x] = {input = {}, output = 0}
    local sample = generatesample()
    for y=1, #sample.input do
        evaldata[x].input[y] = sample.input[y]
    end
    evaldata[x].output = sample.output
end
print("Training...")
model:train(
    {},-- empty dataset
    generatesample,
    0.01,-- target loss, large bc i have no faith in this
    evaldata,
    true
)
print("Training complete, wakey from your nap!")
local function predict(a, b)
    -- it's embarrasing I do this because tanh is a saturating little demon
    local output = model:forward({
        normalize(a),
        normalize(b)
    })
    return denormalize(output)
end
print("\nInference tests:\n")
for i = 1, 10 do
    local a = math.random(0, 10)
    local b = math.random(0, 10)
    local prediction = math.abs(predict(a, b))
    print(string.format(
        "%d + %d = %.2f",
        a,
        b,
        prediction
    ))
end
NN.save(model, "addition_model.four")