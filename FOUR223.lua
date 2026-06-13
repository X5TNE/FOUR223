-- FOUR223
-- An experimental neural-network-and-evolution-inspired model implemented in Lua.

-- This file is a "formalized" and/or cleaner version of FOUR223 for the first version see `nn.lua`.
-- Author: beanboi_223 aka X5TNE, read: https://lunarai.pp.ua/whoami.html
-- Github: https://github.com/X5TNE/FOUR223
-- License: Mozilla Public License

-- I HAVE NOT REIMPLEMENTED THE TRAINING FUNCTION AND MUTATION YET!!!!
-- P.S.: I lowkey miss my chaotic comments but this will allow the model to get more attention (clout i guess rofl) anyway i still have some funny references in here lol

---

-- Helpers

four223 = {}

function four223.aggregate(inputs, weights)
    local summary = 0
    for x=1, #weights do
        local localsum = 0
        for y=1, #inputs do
            localsum = localsum + inputs[y] * weights[x][y]
        end
        summary = summary + localsum
    end

    return summary
end

function four223.randomfloat(min, max)
    if not (min and max) then -- check if both x and y exist, if not **both** we default to these two values:
        min = -1
        max = 1
    end
    return (math.random(min * 1000, max * 1000) / 1000) -- with thousandth precision
end

-- Input Count: Defines how many inputs can be accepted.
-- Layer Count: Defines how many "layers of weights" should be applied to the inputs.
-- Merge Outputs: Defines if your NN will merge everything into a single output early on or not.
function four223.new(inputcount, layercount, mergeoutputs)
    -- ok it looks like i removed all the checks from the last code but i have a point to prove later on
    local returnedmodel = {}
    setmetatable(returnedmodel, { __index = four223.functions })
    returnedmodel.inputbiases = {}
    returnedmodel.weights = {}
    returnedmodel.multipleoutputs = not mergeoutputs

    -- unrelated thing abt pytorch: THE SNAKE COPY HAD STOLE THE FIRE FROM THE TORCH
    for x=1, #layercount do
        local layer = {}
        for y=1, #inputcount do
            layer[y] = four223.randomfloat(-2, 2)
        end
        returnedmodel.weights[x] = layer
    end

    for y=1, #inputcount do
        returnedmodel.inputbiases[y] = four223.randomfloat(-2, 2)
    end

    return returnedmodel
end

function four223.save(model)

-- acts

-- #saturatedlikeghee
function four223.tanh(x)
    if x > 20 then return 1 end
    if x < -20 then return -1 end
    local e = math.exp(2 * x)
    return (e - 1) / (e + 1)
end

-- pretty much standard approximation
function four223.gelu(x)
    local c = 0.7978845608 -- sqrt(2/pi)
    return 0.5 * x * (1 + four223.tanh(c * (x + 0.044715 * x * x * x)))
end

-- the best one ever
function four223.silu(x)
    return x / (1 + math.exp(-x))
end

-- stolen from unformalized version
function four223.softmax_like(scores)
    local maxv = -1e9
    for i = 1, #scores do
        if scores[i] > maxv then maxv = scores[i] end
    end

    local sum = 0
    local exps = {}

    for i = 1, #scores do
        -- stabilize
        local e = math.exp(scores[i] - maxv)
        exps[i] = e
        sum = sum + e
    end

    for i = 1, #exps do
        exps[i] = exps[i] / (sum + 1e-9)
    end

    return exps
end

---

-- the architecture?

four223.functions = {}


function four223.functions.forward(self, inputs)
    -- First we'll bias the inputs, think of this as self-attention just that it's really basic 
    -- and we're getting rid of parts that are usually redundent or not useful.
    local biased = {}
    for i=1, #inputs do
        biased[i] = inputs[i] * self.inputbiases[i]
    end
    
    -- then check for multiple outputs are needed:
    -- if yes, then we'll mimic self-attention + not collapse into a single val
    -- E X P L O S I ON
    if self.multipleoutputs == true then
        -- fake self-attention
        local newinputs = {}
        for x=1, #biased do
            local scores = {}
            for y=1, #biased do
                local interaction = inputs[x] * inputs[y]
                scores[y] = four223.gelu(interaction)
            end
            -- local attention?
            local attn = four223.softmax_like(scores)
            
            local sum = 0
            for y=1, #biased do
                sum = sum + attn[y] * biased[y]
            end
            newinputs[x] = sum
        end

        -- my favorite part
        local output = {}
        for z=1, #newinputs do
            local val = newinputs[z]
            for x=1, #self.weights do
                for y=1, #newinputs do
                    val = val * self.weights[x][y]
                end
            end

            val = four223.silu(val) -- i never actually tested the previous version, this might be better
            output[z] = val
        end
        
        return output
    elseif self.multipleouts == false then
        local mixed = {}
        for x=1, #biased do
            local sum = 0
            for y=1, #biased do
                local interaction = math.abs(biased[x] - biased[y]) / 10 + biased[y] * self.weights[x][y]
                sum = sum + NN.gelu(interaction)
            end
            mixed[x] = sum
            newinput = four223.aggregate(mixed)
            for x=1, #weights do
                for y=1, #mixed do
                    newinput = four223.tanh(newinput * self.weights[x][y])
                end
            end
            return newinput
        end
    end
end

return four223