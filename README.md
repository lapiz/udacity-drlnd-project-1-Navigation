# Project 1 : Navigation

This repository contains an implementation of project 1 for [Udacity's Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## Project Details

In this project, I trained a agent to navigate and collect bananas. According to project introduciton, rewards are provided for collecting a banana. If yellow one will give +1, or blue one -1. The state space has 37 dimensions, and contains velocity, along with ray-based perception of objects around agent's forward direction. Four discrete actions are available : Forward, Backward, Turn Left, Turn Right. The task is episodic.

My agent get an average score of +13 over 100 consecutive episodes.

## Getting Started

### Python environment

- If you run this project on your own environment, you install some packages.
  - Python == 3.6
  - pytorch == 0.4
  - mlagents == 0.4 (Unity ML Agents)
- Or you can run this project on jupyter noteobook.

### Dependencies

To set up your python environment (with conda) to run code in the project, follow the intstruction below.

- Create and activate a new envirionment with Python 3.6

```bash
conda create --name project1 python=3.6
conda activate project1
```

- Clone my project repository and install requirements.txt

```bash
git clone https://github.com/lapiz/udacity-drlnd-project-1-Navigation.git
cd udacity-drlnd-project-1-Navigation
pip install -r requirements.txt
```

### Downloading the Unity environment

Different versions of the Unity environment are required on different operational systems.

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Linux Headless: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
  
## Instructions

- All trained file is saved at results folder
- All tested file is save at test_result folder
- If you want train.
  - python navigation.py train
- Or test already trained data
  - python navigation.py [test count]
- You can change some hyperparameters by editing navigation.py
- On juypter notebook
  - Restart your kernel and run it!

## Files

- README.md
  - This file
- requirements.txt
  - python environment requirements packages
  - Use pip with -r options
- Navigation.ipynb
  - Main notebook file.
  - Based on udacity project skelecton notebook
  - I implemented my agent and some helper classes.
- Report.ipynb
  - My Project report.
  - Include these things
    - Learning Algorithm
      - Hyperpameters
      - Model architechures
    - Plot of Rewards
- model.pt
  - trained model weights
- model.py
  - Model described by two-hidden layer
- dqn_agent.py
  - DQN Agent implementaion based on udacity DQN sample code
- env.py
  - Enviroment wrapper for test
- navigation.py
  - helper codes for training
- scores.py
  - helper code for score data
