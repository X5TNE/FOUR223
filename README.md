uhh forgot to fix this i was gonna but i didn't so here is an AI generated version, be carful because ai is dangerous and can get things really rong:
# FOUR223

An experimental neural-network-and-evolution-inspired machine learning model implemented entirely in Lua.

FOUR223 combines attention-like input mixing, nonlinear activations, mutation, and hill-climbing optimization in a small pure-Lua model.

It is intentionally unconventional and does not follow standard deep-learning architectures.

Inspired by:

* Neural networks
* Self-attention mechanisms
* Evolutionary algorithms
* Hill-climbing optimization

> "If it works, it works."

## Features

* Pure Lua implementation
* No external dependencies
* Single-output and multi-output modes
* Attention-inspired input mixing
* Evolutionary mutation system
* Hill-climbing training step
* Model serialization (`.four` format)
* Copy and mutation utilities
* Multiple activation functions:

  * Tanh
  * GELU
  * SiLU (Swish)

## Installation

Clone the repository:

```bash
git clone https://github.com/X5TNE/FOUR223.git
```

Then require it in your Lua project:

```lua
local four223 = require("four223")
```

## Creating a Model

```lua
local model = four223.new(
    8,      -- input count
    4,      -- layer count
    false   -- merge outputs?
)
```

### Parameters

| Parameter      | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| `inputcount`   | Number of input values                                                      |
| `layercount`   | Number of weight layers                                                     |
| `mergeoutputs` | If `true`, produces a single output. If `false`, produces multiple outputs. |

## Running Inference

```lua
local output = model:forward({
    0.1,
    0.5,
    -0.2,
    0.8
})
```

Outputs are always returned as a Lua table.

Example:

```lua
print(output[1])
```

## Saving Models

```lua
four223.save(model, "example.four")
```

FOUR223 uses a simple text-based `.four` format for serialization.

## Loading Models

```lua
local model = four223.load(
    "example.four",
    8,  -- input count
    4   -- layer count
)
```

## Copying Models

```lua
local clone = four223.copy(model)
```

Creates a deep copy of the model.

## Mutating Models

```lua
local mutated = four223.mutate(model)
```

Mutation slightly perturbs all weights and biases.

This is intended for evolutionary optimization and exploration.

## Training

FOUR223 currently uses a mutation-based hill-climbing optimization strategy.

```lua
model = four223.step(
    model,
    inputs,
    desired_output
)
```

Example:

```lua
model = four223.step(
    model,
    {1, 2, 3},
    {0.75}
)
```

The algorithm:

1. Copies the current model.
2. Generates mutations.
3. Evaluates each mutation.
4. Keeps the mutation with the lowest loss.
5. Returns the best discovered model.

Loss is calculated using absolute error:

```text
loss = Σ |prediction - target|
```

## Architecture Overview

### Input Biasing

Each input is first transformed using:

```text
SiLU(input × bias)
```

### Multi-Output Mode

When outputs are not merged:

* Inputs interact with each other
* Attention-like weights are generated
* Softmax normalization is applied
* Values pass through iterative weight transformations
* Final activation uses:

```text
sin(x)
```

### Single-Output Mode

When outputs are merged:

* Inputs are mixed using learned interactions
* GELU activation is applied
* Values are aggregated
* Multiple weight layers are applied
* Final activation uses:

```text
tanh(x)
```

## Activation Functions

### Tanh

```lua
four223.tanh(x)
```

### GELU

```lua
four223.gelu(x)
```

### SiLU

```lua
four223.silu(x)
```

## File Format

A `.four` file stores:

1. Input biases
2. Weight layers
3. Output mode flag

The format is human-readable and can be inspected with any text editor.

## Current Status

FOUR223 is experimental software.

Implemented:

* Model creation
* Inference
* Mutation
* Hill-climbing optimization
* Save/load support
* Multi-output mode
* Single-output mode

Planned:

* Whatever seems funny or useful at the time
* More experiments
* More weird ideas
* Probably some improvements
* No promises though

## Philosophy

FOUR223 is not intended to compete with frameworks such as PyTorch or TensorFlow.

Instead, it explores the idea that useful behavior can emerge from surprisingly simple mechanisms:

* Input interactions
* Nonlinear transformations
* Evolutionary search
* Iterative refinement

The project embraces experimentation, unconventional design choices, and curiosity-driven machine learning research.

## Author

**beanboi_223 (X5TNE)**

Personal site:

https://lunarai.pp.ua/whoami.html

GitHub:

https://github.com/X5TNE

## License

Mozilla Public License (MPL)

See the LICENSE file for details.
