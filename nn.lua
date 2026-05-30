-- FOUR223
-- Inspired by: Not Google's Transformer
-- Author: beanboi_223 aka X5TNE, https://lunarai.pp.ua/whoami.html
-- Github: https://github.com/X5TNE/FOUR223
-- What I'd call a fake neural network.
-- What someone thought was revolutionary.

-- Don't take any of the comments seriously unless they actually document what each func takes in as input.

-- CODE LICENSE: Mozilla Public License

NN = {}

-- Define "the base"

-- Will be used as the holder for funcs "forward", "train"
NN.base = {}

-- Huh? No I don't think this is dotprd or project.

function NN.project(inputs, weights)
    --local out = {}
    local globalsum = 0
    for o = 1, #weights do
        local sum = 0
        for i = 1, #inputs do
            sum = sum + inputs[i] * weights[o][i]
        end
        globalsum = globalsum + sum
        --out[o] = sum
    end
    --return out
    return globalsum -- combined???
end

-- Normalize funcs
function NN.tanh(x)
    if x > 20 then return 1 end
    if x < -20 then return -1 end
    local e = math.exp(2 * x) -- "^" is bad
    return (e - 1) / (e + 1)
end

function NN.gelu(x)
    -- approximation of GELU (fast version)
    local c = 0.7978845608 -- sqrt(2/pi)
    return 0.5 * x * (1 + NN.tanh(c * (x + 0.044715 * x * x * x)))
end

-- Random                                                                                                                                                                                                   omg guys jake paul here this thing that has four in the name intended a four digit random number thing i think do you guys think this is important 🫨🫨🫨🫨
function NN.randomnum(x, y)
    if not (x or y) then
        x = -1000
        y = 1000
    else
        x = x * 1000
        y = y * 1000
    end
    -- really brute force here, I personally don't like zeros. 
    local number = math.random(x, y) / 1000
    repeat
        number = math.random(x, y) / 1000
    until number ~= 0
    return number
end

-- Define a new, random "NN"

-- Inputs Count: How many inputs are passed in? (Must be more than one.. think of this as weights 🫨)
-- Manipulation Layers Count: How many times should the inputs be manipulated? (Think of this as the more weights... 🤯)
function NN.random(inputscnt, manipulationlayerscnt)
    if inputscnt < 1 or inputscnt > manipulationlayerscnt then error("Input count does not match conditions. -FOUR223 NN / NN.random") end
    local returnednn = {}
    setmetatable(returnednn, { __index = NN.base })
    returnednn.inputlayers = {}
    returnednn.manipulationlayers = {}
    for i=1, inputscnt do
        table.insert(returnednn.inputlayers, NN.randomnum()) -- wannabe bias
    end

    for l = 1, manipulationlayerscnt do
        local layer = {}
        for o = 1, inputscnt do
            layer[o] = NN.randomnum(-2, 2)
        end
        returnednn.manipulationlayers[l] = layer
    end
    return returnednn
end

-- Define the forward pass of the "NN"

-- Inputs: Your inputs as a table ({1, 2, 3, 4, ..})
-- Use Noise: GO BERSERK! #chaos
function NN.base.forward(self, inputs, usenoise)
    if not usenoise then usenoise = false end
    if not self.inputlayers then error("Not self. -FOUR223 NN / NN.base.forward") end
    if #inputs ~= #self.inputlayers then error("Input count does not match conditions. -FOUR223 NN / self:forward") end
    local weights = self.manipulationlayers

    -- talks and bigg biases
    local newoldinputswth = {}

    for i=1, #inputs do
        newoldinputswth[i] = inputs[i] * self.inputlayers[i]
    end

    local mixed = {}
    for x = 1, #inputs do -- we may ignore some stuff this way but whatevs
        local sum = 0
        for y = 1, #inputs do
            local interaction = math.abs(newoldinputswth[x] - newoldinputswth[y]) / 10 + newoldinputswth[y] * weights[x][y]
            sum = sum + NN.gelu(interaction)
        end
        mixed[x] = sum
    end

    local safeinput = NN.project(mixed, weights) -- not sure this would even work :/
    local newinput = safeinput

    local noise = 0

    for x=1, #weights do
        for y=1, #mixed do
            if usenoise == true then noise = (NN.randomnum() / 10) end
            newinput = NN.tanh(newinput * weights[x][y] + noise)
            noise = 0
        end
    end

    return newinput
end

-- Train

-- Dataset: a list of dictionaries of which have "input" and "output"
-- dataset = {}
-- datasetpart = {}
-- -- let's do addition!!!! :D
-- datasetpart.input = {9, 10}
-- datasetpart.output = 21
-- table.insert(dataset, datasetpart)
--
-- -- becomes
--
-- dataset = {{input = {9, 10}, output = 21}}
--
-- Generate Samples: a function you give `train` to generate samples. Will only be used if dataset is empty or a full epoch has passed with model still failing.
-- Let's do addition!!!!
-- function samplefunctionfortrain()
--     local a = NN.random() * 10
--     local b = NN.random() * 10
--     local c = ((((((((a + b) - a) + b) + a) - b) * 0) + a) + b) -- SUPER SMARTS
--     return {input = {a, b}, output = c}
-- end
-- mynn.train({}, samplefunctionfortrain, 0.2)
--
-- -- A few generations and stuff until target loss of < 0.2 is made
-- print(mynn.forward({2, 2})) -- 4 😎
--
-- Target loss: Train script continues until this or lower is reached for at least 1000 forwards
--
-- NN's Favorite Workout Music: You Are an Idiot! by APOCALIPSIS; mentalHAZRD

-- More about this training function:
-- it's just a wrapper

function NN.base.train(self, data, generate, target, evaldata, printloss)
    if not printloss then local printloss = false end
    if not self.inputlayers then error("Not self. -FOUR223 NN / NN.base.forward") end
    if ((not evaldata) and (not generate)) then error("gng the target") end
    if data == nil or #data == 0 then local nodata = true end
    if not generate then local nogenerate = true end
    if nogenerate and nodata then error("No data or generator given. -FOUR223 NN / self:train.") end
    if not target then print("WARNING: Target loss not given, using default 0.5. -FOUR223 NN / self:train.") target = 0.5 end
    
    local count = 0
    local loss = 2^30
    local passeddataset = 0
    -- wait these never even got added???
    local epsilon = 1e-4
    local lr = 0.09

    -- a wild function that took three weeks too long to implement appeared!
    local function fd_step(inputs, truth)
        local base_loss = math.abs(self:forward(inputs) - truth)
        -- input layers
        for i = 1, #self.inputlayers do
            local old = self.inputlayers[i]
            self.inputlayers[i] = old + epsilon
            local lp = math.abs(self:forward(inputs) - truth)
            local grad = (lp - base_loss) / epsilon
            self.inputlayers[i] = old - lr * grad
        end
        -- weights
        for x = 1, #self.manipulationlayers do
            for y = 1, #self.inputlayers do
                local old = self.manipulationlayers[x][y]
                self.manipulationlayers[x][y] = old + epsilon
                local lp = math.abs(self:forward(inputs) - truth)
                local grad = (lp - base_loss) / epsilon
                self.manipulationlayers[x][y] = old - lr * grad
            end
        end
    end

    repeat -- A clever jester had influenced decisions at court just by making the right joke at the right time, but a clown has not. Why? The clown never got the title to appear in the house.
        local sample
        local output
        local inferenceithink
        local truth
        local distanceakalossiguess
        if passeddataset == 25 and nogenerate then error("Training cannot continue. Get a higher quality or larger dataset, or make a generation function. -FOUR223 NN / self:train.") end
        if (nodata and generate)  or (data and generate and passeddataset >= 1) then
            sample = generate()
            inferenceithink = self:forward(sample.input)
            truth = sample.output
            distanceakalossiguess = math.abs(inferenceithink - truth)
            if distanceakalossiguess > target then
                if math.random(1, 4) == 4 then
                    self.mutationtest(sample.input, distanceakalossiguess, target, truth, 2000)
                else
                    fd_step(sample.input, truth)
                end
            end
        else -- mE whEn LE daTasET mAx PaSs of 25 💅👺💅
            for i=1, #data do
                inferenceithink = self:forward(data[i].input)
                truth = data[i].output
                distanceakalossiguess = math.abs(inferenceithink - truth)
                if distanceakalossiguess > target then
                    if math.random(1, 4) == 4 then
                        self.mutationtest(data[i].input, distanceakalossiguess, target, truth, 2000)
                    else
                        fd_step(data[i].input, truth)
                    end
                end
            end
        end
        if (not evaldata) then
            sample = generate()
            inference = self:forward(sample.input)
            truth = sample.output
            dist = math.abs(inference - truth)
            loss = dist
            if printloss == true then print(loss) end
            if dist <= target then count = count + 1 else count = 0 end
        end

        if evaldata then
            for i=1, #evaldata do
                sample = evaldata[i]
                inference = self:forward(sample.input)
                truth = sample.output
                dist = math.abs(inference - truth)
                loss = dist
                if printloss == true then print(loss) end
                if dist <= target then count = count + 1 else count = 0 end
            end
        end
    until loss <= target and count >= 1000 -- ok but class of 2020 still didnt get a graduation 
end

-- Intended to be used through the high level func `{yournn}.train`.
-- **I mean it isnt backprop., but I like it this way**
-- Don't mind the naming.
-- "stochastic hill climbing / evolution strategy"
function NN.base.mutationtest(self, inputs, originalloss, targetloss, truth, maxtries)
    local function lazyahhcopy(neuraln)
        local currself = NN.random(#neuraln.inputlayers, #neuraln.manipulationlayers)
        for x=1, #neuraln.manipulationlayers do
            for y=1, #neuraln.inputlayers do
                currself.manipulationlayers[x][y] = neuraln.manipulationlayers[x][y]
            end
        end
        for y=1, #neuraln.inputlayers do
            currself.inputlayers[y] = neuraln.inputlayers[y]
        end
        return currself
    end
    local function curse(currself)
        for x=1, #currself.manipulationlayers do
            for y=1, #currself.inputlayers do
                currself.manipulationlayers[x][y] = currself.manipulationlayers[x][y] + (NN.randomnum() / 150) -- smooth criminal
            end
        end
        for y=1, #currself.inputlayers do
            currself.inputlayers[y] = currself.inputlayers[y] + (NN.randomnum() / 150) -- btw did yall notice hidden six seven
        end
    end
    local bestloss = originalloss
    local bestver = lazyahhcopy(self) -- [insert gif of skeleton falling]
    for i=1, maxtries do
        local politician = lazyahhcopy(self) -- make a clown
        curse(politician) -- make the clown do something different from other clowns
        local claim = politician.forward(politician, inputs) -- ask the clown a question
        local loss = math.abs(claim - truth)
        if loss < bestloss then
            bestver = lazyahhcopy(politician)
            bestloss = loss
        end
    end

    self.inputlayers = bestver.inputlayers
    self.manipulationlayers = bestver.manipulationlayers
end

-- Was intended to be used through the high level func `{yournn}.train`. 
--[[
function NN.base.bruteforcedecrease(self, inputs, origloss)
    error("Should not be ran!! Old function!!")
    return
    local newself = {}
    newself.inputlayers = {}
    newself.manipulationlayers = {}
    -- mmm chicken tenders 😍
    local potentialcon_tenders = {}
    setmetatable(newself, { __index = NN.base })
    setmetatable(newself.inputlayers, { __index = self.inputlayers })
    setmetatable(newself.manipulationlayers, { __index = self.manipulationlayers })
    
    -- first we do changing the bias of the inputs
    for x=1, #newself.inputlayers do
        local finalout = {}
        finalout.isinputlayer = true
        local posbiaschange = 0
        local negbiaschange = 0
        local lastchange = 0
        newself.inputlayers[x] = self.inputlayers[x] + 0.004 -- smol change to see how loss changes                                                                                                                                                                            jake paul here again and OMG is that four again?!?!?!
        posbiaschange = (newself:forward(inputs) - origloss)
        newself.inputlayers[x] = self.inputlayers[x] - 0.004
        negbiaschange = (newself:forward(inputs) - origloss)
        -- compare and find how far loss goes down and ok maybe this is closer to evolutionary search but that doesn't matter "neural" sounds cool
        if negbiaschange < posbiaschange and negbiaschange < math.abs(negbiaschange) then 
            finalout.inputlayer = {x, 1}
            finalout.type = -1
            while negbiaschange < lastchange do
                lastchange = negbiaschange
                newself.inputlayers[x] = newself.inputlayers[x] - 0.001
                negbiaschange = (newself:forward(inputs) - origloss)
            end
            finalout.mostloss = lastchange
            table.insert(potentialcon_tenders, finalout)
        end
        if posbiaschange < negbiaschange and posbiaschange < math.abs(posbiaschange) then 
            finalout.inputlayer = {x, 1}
            finalout.type = 1
            while posbiaschange < lastchange do
                lastchange = posbiaschange
                newself.inputlayers[x] = newself.inputlayers[x] + 0.001
                posbiaschange = (newself:forward(inputs) - origloss)
            end
            finalout.mostloss = lastchange
            table.insert(potentialcon_tenders, finalout)
        end
    end

    -- now we do changing the manipulation layer weights slightly
    for x=1, #self.manipulationlayers do
        for y=1, #self.inputlayers do
            -- reusing code
            local finalout = {}
            finalout.isinputlayer = false
            local posbiaschange = 0
            local negbiaschange = 0
            local lastchange = 0
            newself.manipulationlayers[x][y] = self.manipulationlayers[x][y] + 0.004
            posbiaschange = (newself:forward(inputs) - origloss)
            newself.manipulationlayers[x][y] = self.manipulationlayers[x][y] - 0.004
            negbiaschange = (newself:forward(inputs) - origloss)
            if negbiaschange < posbiaschange and negbiaschange < math.abs(negbiaschange) then 
                finalout.inputlayer = {x, y}
                finalout.type = -1
                while negbiaschange < lastchange do
                    lastchange = negbiaschange
                    newself.manipulationlayers[x][y] = newself.manipulationlayers[x][y] - 0.001
                    negbiaschange = (newself:forward(inputs) - origloss)
                end
                finalout.mostloss = lastchange
                table.insert(potentialcon_tenders, finalout)
            end
            if posbiaschange < negbiaschange and posbiaschange < math.abs(posbiaschange) then 
                finalout.inputlayer = {x, y}
                finalout.type = 1
                while posbiaschange < lastchange do
                    lastchange = posbiaschange
                    newself.manipulationlayers[x][y] = newself.manipulationlayers[x][y] + 0.001
                    posbiaschange = (newself:forward(inputs) - origloss)
                end
                finalout.mostloss = lastchange
                table.insert(potentialcon_tenders, finalout)
            end
        end
    end

    return potentialcon_tenders -- and i think this is the end of the longest func in the script :O used later on in self:train to get the four best contenders, do some matchmaking, find the best group or pair or single
end
]]

-- add me on polytoria guys: https://polytoria.com/u/beanboi223
-- remember skids: your model weights are not licensed under MPL yknow, it's just this lua script file. do whatever you want with the model weights, i don't endorse what you do specifically tho.
-- script prob doesn't even work, i just wrote it without testin.
-- lua 5.4
-- 1.0 finished: 5/28/26

-- omg wild save and load function was just made 5/29/26

function NN.save(model, filename)
    local f = assert(io.open(filename, "w"))
    -- inputlayers "the trench"
    for i = 1, #model.inputlayers do
        f:write(model.inputlayers[i] .. " ")
    end
    f:write("\n===\n")
    -- manipulation layers "no man's land"
    for l = 1, #model.manipulationlayers do
        for i = 1, #model.inputlayers do
            f:write(model.manipulationlayers[l][i] .. " ")
        end
        f:write("\n")
    end
    f:close()
end

function NN.load(filename, input_size, manipulation_layers_count)
    local f = assert(io.open(filename, "r"))
    local model = NN.random(input_size, manipulation_layers_count)
    -- load inputlayers
    local line = f:read("*l")
    local i = 1
    for num in string.gmatch(line, "([^%s]+)") do
        model.inputlayers[i] = tonumber(num)
        i = i + 1
    end
    -- load manipulation layers
    for l = 1, manipulation_layers_count do
        local line = f:read("*l")
        if line == "===" then line = f:read("*l") end -- next line
        local j = 1
        for num in string.gmatch(line, "([^%s]+)") do
            -- guard just in case file is malformed
            if j <= #model.inputlayers then
                model.manipulationlayers[l][j] = tonumber(num)
            end
            j = j + 1
        end
    end
    f:close()
    return model
end

-- oh yeah and return the table bc this is a module
return NN