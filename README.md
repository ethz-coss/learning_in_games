# Learning in Games
This package contains games and learning methods implemented as python functions. The package was created to study the dynamics of Q-learning in congestion games, and was later expanded to include a series of other settings from game theory.

## Citing
The code in this repository was used to produce the following paper which investigated the effects of Q-learning in the Braess Paradox, and found some counter-intuivtive behaviour for the exploration rate of Q-learners which led to positive welfare effects which could escape the worst Price of Anarchy.

```
@article{carissimo2024counter,
  title={Counter-intuitive effects of Q-learning exploration in a congestion dilemma},
  author={Carissimo, Cesare},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}
```

## Acknowledgment
This work was funded by European Union’s Horizon 2020 Research and Innovation Programme through the Distributed Intelligence & Technology for Traffic & Mobility Management ([DIT4TraM](https://dit4tram.eu/)) (Grant Number: 953783)

## Contents
### Games
- Routings Games
  - Initial Braess Network
  - Augmented Braess Network
  - Pigou Network
- Prisoners Dilemma
- Minority Game
- El Farol Bar
- Pupulation Game
- Duopoly Pricing Game
- Public Goods Game

### Reinforcement Learning Algorithms
- Bellman Update Rule
- epsilon greedy action selection
- boltzman action selection (smooth Q-learning)


### Notes
what are the features of the package that are most important?
- the many implemented environments specific to game theory
- the implementation of Q-learning with easy extensions of its variants
- implementation of all functions as vectorized NumPy operations


### steps to make this a package
- [x] use pip requirements, and make sure only the necessary requirements are present
- [x] use dataclasses to achieve modularity, where possible, needed for:
  - run simulation code to know which parameters it needs, e.g.
    - duopoly requires states but congestion does not
    - public goods game takes a multiplier
    - network congestion games take parameters for each edge
  - multiprocessing sweeps to know necessary parameters
  - plotting of relevant variables and parameters after simulation
- [x] documentation of important functions
  - game functions and dataclasses
  - agent functions
  - plotting functions
  - other functions
- [ ] plotting functions
  - welfare over time
  - action distribution over time
  - vector field plot, simplex
  - q values plots
- [x] setuptools toml file
- [x] package structure
- [ ] tutorial notebooks for running games
- [ ] final packaging steps 
  - [ ] run black code formatter
  - [ ] try flake8
  - [ ] use twine for publishing to pypi
  - [ ] sign up and upload to pypi
- [ ] remove old committed files from git, and transition to main branch as center for the package
