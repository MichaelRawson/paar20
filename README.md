# Materials for black-box ATP reinforcement learning.

This repository contains some code and materials we used during production of the accompanying paper.
They are provided on an as-is basis and will not work without some effort: the technique entails a significant amount of technical complexity and some amount of non-determinism.
That said, we look forward to hearing from others attempting derived work!

## Layout
In no particular order:

- `atp.py` contains code used for running a specific configuration of Vampire as described in the paper
- `mcts.pt` is a quick implementation of monte-carlo tree search, used for data generation
- `graphs.py` contains useful graph manipulation routines, including row-normalised adjacency matrices reasonably efficiently (for which I apologise...).
- `model.py` contains the neural network implementation.
- `Cargo.*`, `src/lib.rs` is a Rust parser for various fragments of TPTP which we call into from the Python scripts. It also transforms said fragments into graphs in a specific format.
- `hol_problems` contains some Church-numeral training problems in THF.
- `scripts/generate` generates training data given a problem
- `scripts/{random,model}-policy` have doubled up for different evaluations, but they either load the model and run the policy or select actions randomly.
- `results/` contains the first-order results from GRP001-1 and all of GRP. Data files are not included due to size.

## Workflow
To run an experiment as described in the paper, it might look a little like this:
- Pick a set of training and test problems.
- Run `scripts/generate` on each training problem to generate a whole heap of data.
- Run `scripts/train` to train a model on said heap-o-data. Progress is plotted in TensorBoard.
- Run `scripts/{random,model}-policy` on the test set to get your results.
- Scratch your head.
- Tweak some parameters or the algorithm, try again...

## Off-the-shelf RL
Older commits in this repository implement some sort of off-the-shelf RL instead of our newer method.
Commits typically represent days where it was working, which it very rarely does.
If you are an RL expert, have a laugh at our expense and implement it yourself.
If you are not: here be dragons.
