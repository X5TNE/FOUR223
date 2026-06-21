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
    for x=1, layercount do
        local layer = {}
        for y=1, inputcount do
            layer[y] = four223.randomfloat(-2, 2)
        end
        returnedmodel.weights[x] = layer
    end

    for y=1, inputcount do
        returnedmodel.inputbiases[y] = four223.randomfloat(-2, 2)
    end

    return returnedmodel
end

-- stolen from uncivilized version
function four223.save(model, filename)
    local f = assert(io.open(filename, "w"))
    -- inputlayers "the trench"
    for i = 1, #model.inputbiases do
        f:write(model.inputbiases[i] .. " ")
    end
    f:write("\n===\n")
    -- manipulation layers "no man's land"
    for l = 1, #model.weights do
        for i = 1, #model.inputbiases do
            f:write(model.weights[l][i] .. " ")
        end
        f:write("\n")
    end
    -- the flag
    if model.multipleoutputs == true then
        f:write("1")
    else
        f:write("0")
    end
    f:close()
end

-- stolen from uncivilized version
-- i call it the .four filetype
function four223.load(filename, inputcount, layercount)
    local f = assert(io.open(filename, "r"))
    local model = four223.new(inputcount, layercount, true)
    -- load inputlayers
    local line = f:read("*l")
    local i=1
    for num in string.gmatch(line, "([^%s]+)") do
        model.inputbiases[i] = tonumber(num)
        i = i + 1
    end
    -- load manipulation layers
    for l=1, layercount do
        local line = f:read("*l")
        if line == "===" then line = f:read("*l") end -- next line
        local j = 1
        for num in string.gmatch(line, "([^%s]+)") do
            -- guard just in case file is malformed
            if j <= #model.inputbiases then
                model.weights[l][j] = tonumber(num)
            end
            j = j + 1
        end
    end
    line = f:read("*l")
    if line == "1" then model.multipleoutputs = true else model.multipleoutputs = false end
    f:close()
    return model
end

-- how to copy a model
function four223.copy(model)
    local newmodel = {}
    setmetatable(newmodel, { __index = four223.functions })
    newmodel.inputbiases = {}
    newmodel.weights = {}
    newmodel.multipleoutputs = model.multipleoutputs
    for x=1, #model.weights do
        local layer = {}
        for y=1, #model.inputbiases do
            layer[y] = model.weights[x][y]
        end
        newmodel.weights[x] = layer
    end
    for y=1, #model.inputbiases do
        newmodel.inputbiases[y] = model.inputbiases[y]
    end
    return newmodel
end

-- I never would've thought of this, but let's make it cleaner.
-- btw this is lowkey just the copy func but we slightly change all the weights and biases.
function four223.mutate(model)
    local newmodel = {}
    setmetatable(newmodel, { __index = four223.functions })
    newmodel.inputbiases = {}
    newmodel.weights = {}
    newmodel.multipleoutputs = model.multipleoutputs
    for x=1, #model.weights do
        local layer = {}
        for y=1, #model.inputbiases do
            -- slight change can make a huge difference, like a dish with a bit too much salt or pepper. 👨‍🍳👌😘
            layer[y] = model.weights[x][y] + (four223.randomfloat() / 10)
        end
        newmodel.weights[x] = layer
    end
    for y=1, #model.inputbiases do
        newmodel.inputbiases[y] = model.inputbiases[y] + (four223.randomfloat() / 10)
    end
    return newmodel
end

-- four223 training step 
-- btw this is sorta expensive, if you don't like it, make your own.
-- basically this mutates the model a bunch and keeps the best where loss went down the most. 
-- ig it's hill climbing but i don't like that name cuz it sounds like we search through smooth space.
function four223.step(model, inputs, desiredoutput, maxmutations)
    -- now this supports multipleoutputs btw
    local bestmodel = four223.copy(model)
    local bestoutput = bestmodel:forward(inputs)
    local bestloss = 0
    for i=1, #bestoutput do
        bestloss = bestloss + math.abs(bestoutput[i] - desiredoutput[i])
    end
    if not maxmutations then maxmutations = (#model.inputbiases * #model.weights)^2 end -- risky and slow but worth it tbh
    for m=1, maxmutations do
        local currentmodel = four223.mutate(bestmodel)
        local output = currentmodel:forward(inputs)
        local loss = 0
        for i=1, #output do
            loss = loss + math.abs(output[i] - desiredoutput[i])
        end
        if loss < bestloss then 
            bestmodel = four223.copy(currentmodel)
            bestloss = loss
        end
    end
    return bestmodel
end

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

-- the best one evers
function four223.silu(x)
    return x / (1 + math.exp(-x))
end

-- stolen from the uncivilized version
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
    -- and we're getting rid of parts that are **usually** redundent or not useful.
    -- In image generation this can actually be really useful as we can ignore parts people don't really look at.
    -- (like away from the center subject, or around the corners of the image.)
    local biased = {}
    for i=1, #inputs do
        biased[i] = four223.silu(inputs[i] * self.inputbiases[i]) -- hmm
    end
    
    -- then check for multiple outputs are needed:
    -- if yes, then we'll mimic self-attention + not collapse into a single val
    if self.multipleoutputs == true then -- the two branches: 3D heaven and collapsing hell
        -- fake self-attention
        local newinputs = {}
        for x=1, #biased do
            local scores = {}
            for y=1, #biased do
                local interaction = biased[x] * biased[y]
                scores[y] = four223.gelu(interaction) -- this right here, ai totally, im not lying and im not joking, it's artifactically generated using four223, which powers a thing called idiotbabel, this def is not a joke
            end
            -- local attention?
            local attn = four223.softmax_like(scores)
            
            local sum = 0
            for y=1, #biased do
                sum = sum + attn[y] * biased[y]
                newinputs[x] = sum / (#biased + 1e-9) -- no   E X P L O S I ON
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
                    val = val / (1 + math.abs(val)) -- no   E X P L O S I ON
                end
            end

            val = math.sin(val) -- i never actually tested the previous version, this might be better
            output[z] = val
        end
        
        return output
    elseif self.multipleoutputs == false then
        local mixed = {}
        for x=1, #biased do
            local sum = 0
            for y=1, #biased do
                local interaction = math.abs(biased[x] - biased[y]) / 10 + biased[y] * self.weights[x][y]
                sum = sum + four223.gelu(interaction)
            end
            mixed[x] = sum
        end
        local newinput = four223.aggregate(mixed, self.weights)
        for x=1, #self.weights do
            for y=1, #mixed do
                newinput = newinput * self.weights[x][y]
            end
            newinput = math.tanh(newinput) -- might be better 👀
        end
        -- sorta dumb but if it works i guess it works, if not, don't use this branch.
        local returnedoutputig = {}
        returnedoutputig[1] = newinput
        return returnedoutputig
    end
end

return four223