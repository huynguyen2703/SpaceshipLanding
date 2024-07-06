"""
 Spaceship Landing is an artificial intelligence program designed to train a spaceship to land on the moon in a
 virtual environment provided by gymnasium library. The goal is to design a deep Q learning neural network to train
 a spaceship to land on a specific location that has been specified in the environment. The details of the steps are
 below :

 For this program, we require specific modules and packages to help us implement a neural network. We will use the
 random module to generate randomness between 0 and 1 to help activate the power of epsilon greedy selection policy.
 This selection policy will help the model choose a specific action based on the comparison between the random number
 and the epsilon value itself. Next, we will need the help from the numpy module to work with vectors and alot of
 computations such as mean, max. Most importantly, now we will need the help from Pytorch library to help us implement
 our artificial neural network with the predefined modules. Then, we need to use the optim submodule from pytorch,
 which will help us create an optimizer, it will apply the gradient descent (or stochastic gradient descent) algorithms
 to update the weights after the loss has been calculated, the goal here is to minimize the value of the cost function
 (the loss), the smaller the value, the closer to the real value the predicted output value will be when compared to the
 target. Next, we will definitely need the submodule that comes from the nn module(inside pytorch library) to implement
 an activation function for our hidden layers; this activation function allows our model to learn complex patterns that
contain non-linearity (the activation function we are going to use is rectifier function). We will also need the
submodule autograd which will compute the gradient (the rate of change) of the loss, they tell us how much the weight
and biases contribute to the total loss so that the optimizer can update those. Though not explicitly used, pytorch
uses automatic differentiation techniques to calculate gradients. We then we need a data structure to store our reward
that the model gets for each training episode. Lastly, we need to import the gymnasium library to create the
environment that the model is going to operate inside.


 In the initialization phase, we initialize a starting learning rate (alpha in Q-learning)
  very small, so that each time we update our weight, the model can cover a small percentage of loss carefully.
  Next, we use something called a minibatch sampling technique, by specifying a minibatch size, because we are going to
  take advantage of both stochastic gradient descent and gradient descent for our model. Next, of course, we need a
  discount factor (y in bellman equation) to help the model balance the importance of immediate rewards and future rewards.
  Then, we had replay_buffer_size, which is the model's memory to support experience replay, which helps improve the
  training process and the learning of the model. Lastly, the interpolation_parameter is used in the updating phase
  related to weights of the local network and the target network (basically for experience replay)

  Next, we would create an environment for the model to train, we will use the make() from gymnasium to create the
  environment we like, and then we have the state shape, which is an 8-dimensional vector, we then have state_size, which
  represents the number of elements in the input layer, then the number of actions would be four, which are the possible
  actions the model can make in a specific state that will result in the model ending in a new state, getting a new
  reward.

  Building Neural Network Phase
  In this phase, we will have three classes to help us complete a neural network; the first class is Network. We will
  make this Network class a submodule of the nn.Model from Pytorch through inheritance to build a neural network, a
  Network instance will take dependency injection, which are the state_size and the action+size, we provide the seed
  to be 42 so that we have the reproducibility when running the code multiple times. Next, we will build each layer
  of the neural network. The first layer is fc1, we take the number of inputs to construct the input layer, and we will
  have 64 neurons for the next layer (the first hidden layer). Next is fc2, we will specify the number of neurons in the
  previous layer, and we want to build the second layer, which also has 64 neurons. Finally, fc3 is the final fully
  connected layer of the neural network, this is where we specify the 64 neurons of the previous layer and the number of
  output neurons, which will be four (each represent an action), then based on the output, we will run the selection
  policy to determine which action will be chosen. The first and only method in the Network is the forward function,
  we use this function for forward propagation, we will first retrieve the fully connected layer and then for each layer
  we retrieve, we will apply the rectifier activation function. Then we will return the final fully connected layer which
  contains the actions. Next, we will have class ReplayMemory to help us perform experience replay; the constructor of
  this class takes capacity as input, which specifies the maximum number of experiences the memory can hold. Self.device
  is set to either "cuda:0" (if a GPU is available) or "cpu". This determines whether the memory will be stored on the
  GPU (for faster processing) or the CPU. We then create a memory list will be used to store the experience. The first
  function in this class is push(), this function takes an event, which represents an experience. Just like humans, we
  are likely to forget the least recent experience and remember the most recent, so if the memory size goes over the capacity,
  we will delete the least recent experience and push the most recent experience. Our next function in this class is the
  sample method it takes a batch_size as input and takes a sample of experiences with batch_size from the memory list; for
  this, we will use the sample() of the random module. It iterates through the sampled experiences and extracts the
  individual components, every component is converted into pytorch tensor, especially the done component need to be converted
  to numpy uint8 data for boolean representation, and then we move to the chosen device, finally we return a tuple of
  all component, each component is a stack. Sampling experiences from the replay memory helps break correlations that
  might exist in consecutive experiences and improves the generalization of the learned Q-values. The final class we have
  is Agent, this is our model, in this class, we create an instance of the Network class (defined earlier) representing
  the local Q-network for action selection. Then we create another instance of the Network class representing the
  target Q-network for training the local network. We need both of these networks for training purposes. We then
  create an Adam optimizer (optim.Adam) to update the weights of the local Q-network during training. We then create an
  instance of the ReplayMemory class (defined earlier) to store experiences for replay. Finally, initializing a counter
  for learning updates (explained later in step). Now, the class contains a total of three methods, step() is used to help
  the model decide when to learn, in this case, the model will accumulate the experiences and start learning every four steps.
  Finally, the model will learn when it accumulates enough data for a minibatch. Next, for the act(), the act function
  takes the current state as input and determines the agent's action. It first converts the state to a format suitable
  for the neural network and puts the network in prediction mode. Then, it calculates the Q-values (estimates of future rewards)
  for all possible actions. The agent leverages an exploration-exploitation trade-off: with a high probability
  (greater than local_epsilon), it chooses the action with the highest Q-value (exploitation for optimal reward).
  However, with a small probability (local_epsilon), it takes a random action (exploration to discover new information
  about the environment). This balance between maximizing immediate rewards and exploring the environment is crucial for
  successful learning. In this case, update state as torch tensor, add an extra dimension to specify which batch
  the state belongs to, and then we convert the local network to evaluation mode to get the Q-values, then we back to
  the training mode, now, this is where selection policy comes in, we will use epsilon greedy to select the action,
  to do that, we use the random() from the random module to generate a number, and we compare that number to epsilon,
  Q-value will be chosen based on the comparison. The argmax function is used to find the index of the element in the
  action_values tensor that has the highest value. This index corresponds to the action that has the highest estimated
  future reward. Otherwise, we will use the choice() from random module for exploration. The learn function takes a
  batch of experiences (experiences) and performs the training process for the DQN agent. We first extract the experiences
  into components, then we compute the Q-values for the next states from the target network, we then call detach(), by
  doing this, we separate the results from the computation graph. The computation graph tracks the operations performed
  on tensors during forward pass calculations. Gradients are calculated based on this graph during backpropagation to
  update network weights. Detaching the result prevents gradients from being calculated for the target Q-values. The reason
  we want to do this is to minimize the instability of the Q-values in the local network and the target network. The reason
  that detaching is crucial is to prevent gradients from being calculated during back propagation; we want to prevent this
  so that the weights of the target network will not be updated, hence we can maintain stability because what we want to
  update is the local network. Next, .max(1) finds the maximum Q-value for each next state across all possible actions (dimension 1).
[0] selects the first element (the maximum Q-value itself).
.unsqueeze(1) adds an extra dimension of size 1 at the beginning. This ensures the target Q-values have the same shape
(batch size, 1) as the expected Q-values from the local network, making them compatible for calculating the loss function later.
Now, after we calculate the next Q-values (Q(s',a')), now we can calculate Q(s,a) by taking the discount factor to multiply
the next Q-values. 1 - dones basically will cancel out the factor and only keep the highest reward, this to indicate there
are no future steps to make, the 1 - dones term ensures that the target Q-values only account for future rewards when
the episode continues, focusing solely on the immediate reward when the episode has ended. Now, we can calculate the loss
when we have q_expected and q_target. Then we will feed the information back to the neural network using backpropagation.
Then, we use the optimizer to update the weight of the network through step(). Finally, we use soft_update() to update
the target network. Inside, soft_update(), A soft update addresses this by gradually updating the target network towards
 the local network. It uses an interpolation parameter (local_interpolation_parameter) between 0 and 1 to control the
 update rate. It gradually merges them using an interpolation parameter.
This allows the target network to incorporate some of the learning progress from the local network, potentially leading
to better target Q-values in the long run and maintain some stability by not adopting the potentially noisy or unstable
weight changes from the local network too quickly. Basically, the formula used in soft_update() is to gradually update and
update the weights in the target network by merging the influence in the local network and still maintain the stability
 of the target network's weights.

 Model Initialization Phase
 Now, we will create an agent with a complete neural network by initializing Agent() with the number of states and number
 of actions.

 Training Phase
 Finally is the training phase, here we will represent the training process by episodes, we aim to train the model in
 2000 episodes. We have 1000 timesteps for each episode, which means the model can make 1000 steps each episode. Next,
 we prepare a starting epsilon value to be 1, and we set the decay to be 0.995, so that the decrease is small, the ending
 value is 0.01, that's the smallest. Finally, we want to accumulate the reward for each 100 episodes in a deque.
 Now is the training, we iterate from 1 to 2000, and each episode we need to reset the environment, reset the reward,
 then we iterate from 0 to 1000 (to see how many steps the model makes) per episode, first we get the action, then we perform
 the action using env.step(action), this results in the model going into a new state, get a new reward and a boolean value of
 done. Then we want the model to learn or accumulate experiences using agent.step() and provide all necessary information.
 Then, model state will be updated, and reward will be updated, if done, then we go to the next episode, we then accumulate
 reward for that episode and determine the current epsilon. Next we will print the average reward for 100 episodes, every time
 100 episodes are done, we will stop and print a new batch of 100 average values, keeping the previous average for the previous
 100 episodes. We know that the model will win if the reward it gets is 200, so as long as the average reward for 100 episodes
 is over 200, the training will stop, we will save our trained local network, and break the training loop, ending our training
 on an artificial neural network.
"""

# import needed libraries
import os  # for random variable generators
import random  # for randomness
import numpy as np  # for matrices and arrays
import torch  # pytorch for deep learning
import torch.nn as nn  # neural network package
import torch.optim as optim  # optimizer used to apply optimization algorithms
import torch.nn.functional as F  # activation functions
import torch.autograd as autograd  # gradient descent
from torch.autograd import Variable  # torch variables
from collections import deque, namedtuple  # data structure to store needed data at the end
import gymnasium as gym  # gymnasium for the environment

# Initializing the hyperparameters
learning_rate = 5e-4  # 5*10^-4
minibatch_size = 100  # good batch size for deep Q-learning
discount_factor = 0.99  # agent will care about long-term cumulative reward
replay_buffer_size = 100000  # agent memory to improve the training process
interpolation_parameter = 0.001  # for experience replay

# Setting up the environment
env = gym.make('LunarLander-v2')  # create environment
state_shape = env.observation_space.shape  # a vector of 8 inputs
state_size = env.observation_space.shape[0]  # number of elements in the input state
number_actions = env.action_space.n  # number of actions agent can perform

print('State Shape: ', state_shape)
print('State Size: ', state_size)
print('Number of Actions: ', number_actions)


# Neural network architecture
class Network(nn.Module):
    def __init__(self, local_state_size, action_size, seed=42) -> None:  # seed = 42 for randomness
        super(Network, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(local_state_size,
                             64)  # first full connection between input layer and the first fully connected layer, we put in the # of neurons for the input layer and the first hidden layer
        self.fc2 = nn.Linear(64, 64)  # the second full connection layer
        self.fc3 = nn.Linear(64, action_size)  # last full connection between the previous layer and the output layer

    def forward(self, local_state):
        x = self.fc1(local_state)  # return the first fully connected layer to x
        x = F.relu(x)  # apply the rectifier activation function

        x = self.fc2(x)  # return the second fully connected layer to x
        x = F.relu(x)  # apply the rectifier activation function

        x = self.fc3(x)  # return the last fully connected layer to x
        return x  # finish forward propagating the signal from the input layer containing the state to the output layer containing the action


class Replaymemory(object):
    def __init__(self, capacity):  # capacity for memory
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")  # may use GPU to make training faster
        self.capacity = capacity  # maximum size of memory buffer
        self.memory = []

    def push(self, event):  # push an experience to memory
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):  # randomly select a batch of experiences from memory buffer
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(
            self.device)  # convert a numpy array into pytorch tensor, change to float and move to a computing device
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(
            self.device)  # convert a numpy array into pytorch tensor, change to long and move to a computing device
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(
            self.device)  # convert a numpy array into pytorch tensor, change to float and move to a computing device
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(
            self.device)  # convert a numpy array into pytorch tensor, change to float and move to a computing device
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)  # convert to boolean using np.uint8 then convert a numpy array into pytorch tensor, change to bool and move to a computing device
        return states, next_states, actions, rewards, dones


class Agent:
    def __init__(self, local_state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = local_state_size
        self.action_size = action_size
        self.local_qnetwork = Network(local_state_size, action_size).to(
            self.device)  # local network to select the action
        self.target_qnetwork = Network(local_state_size, action_size).to(
            self.device)  # target network will calculate the Q-values that will be used in training of the local network
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(),
                                    lr=learning_rate)  # weight of the network and learning rate
        self.memory = Replaymemory(replay_buffer_size)  # memory for experience replay
        self.time_step = 0

    def step(self, local_state, local_action, local_reward, local_next_state, local_done):  # decide when to learn
        self.memory.push((local_state, local_action, local_reward, local_next_state, local_done))
        self.time_step = (self.time_step + 1) % 4
        if self.time_step == 0:
            if len(self.memory.memory) > minibatch_size:  # we learn by minibatch not by observation per observation
                experiences = self.memory.sample(100)
                self.learn(experiences, discount_factor)

    def act(self, local_state, local_epsilon=0.):  # decide action after learning
        local_state = torch.from_numpy(local_state).float().unsqueeze(0).to(
            self.device)  # update state as torch tensor, add an extra dimension to specify which batch the state belongs to
        self.local_qnetwork.eval()
        with torch.no_grad():  # make sure gradient computation is disabled and make sure we are in predict mode
            action_values = self.local_qnetwork(
                local_state)  # get Q-values instances(argument) works because Network inherits from nn.Module -> forward get called implicitly
        self.local_qnetwork.train()  # back to training mode
        if random.random() > local_epsilon:
            return np.argmax(
                action_values.cpu().data.numpy())  # send to cpu and use data.numpy because argmax expects numpy format of data
        return random.choice(np.arange(
            self.action_size))  # if random number > epsilon -> choose action with highest Q-value, else choose a random action

    def learn(self, experiences, local_discount_factor):
        states, next_states, actions, rewards, dones = experiences
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(
            1)  # detach the result from the computation graph (we won't track gradients for this tensor during backpropagation) then get the max(1) returns 2 tensors and[0] to get the maximum then unsqueeze to add dimension of the batch at position 1
        q_targets = rewards + (local_discount_factor * next_q_targets * (1 - dones))
        q_expected = self.local_qnetwork(states).gather(1,
                                                        actions)  # get the q_expected by gathering all the Q-values returned by local_qnetwork
        loss = F.mse_loss(q_expected, q_targets)  # calculate the cost function
        self.optimizer.zero_grad()  # zeroing out the gradient to prevent mixed of gradients in training
        loss.backward()  # backpropagation
        self.optimizer.step()  # update the weight of the network
        soft_update(self.local_qnetwork, self.target_qnetwork,
                    interpolation_parameter)  # update the target network


def soft_update(local_model, target_model, local_interpolation_parameter):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(local_interpolation_parameter * local_param.data + (
                1.0 - local_interpolation_parameter) * target_param.data)  # update the target network parameters


# Initializing the DQN agent
agent = Agent(state_size, number_actions)

# Training the DQN agent
number_episodes = 2000  # total 2000 days to train
maximum_number_timesteps_per_episode = 1000  # 1000 steps per episode
epsilon_starting_value = 1.0  # we start with epsilon value of 1
epsilon_ending_value = 0.01  # the lowest is 0.01
epsilon_decay = 0.995  # decay by 0.995 every episode
epsilon = epsilon_starting_value
rewards_on_100_episodes = deque(maxlen=100)  # we use a deque to store a list of avg award every 100 episodes

for episode in range(1, number_episodes + 1):  # we loop through 2000 episodes
    state, _ = env.reset()  # reset environment for every episode
    agent_reward = 0  # reset reward too
    for t in range(0, maximum_number_timesteps_per_episode):  # loop through every step in an episode
        action = agent.act(state, epsilon)  # agent decides an action first
        next_state, reward, done, _, _ = env.step(action)  # after action is made, it receives a reward, a new status whether episode is done or not and ends up in a new state
        agent.step(state, action, reward, next_state,done)  # then agent tries
        # to learn by taking every component of an experience
        # (step() only works if there are 100 experiences,
        # otherwise, it just adds one more experience to memory
        state = next_state  # then go to new state
        agent_reward += reward # accumulate award
        if done:  # if an episode is done, move on to the next one
            break
    rewards_on_100_episodes.append(agent_reward) # add to deque
    epsilon = max(epsilon_ending_value, epsilon_decay * epsilon) # update epsilon value
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(rewards_on_100_episodes)), end="")
    if episode % 100 == 0:  # print average reward every 100 episodes
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(rewards_on_100_episodes)))
    if np.mean(rewards_on_100_episodes) >= 200.0:
        print('\nEnvironment solved in {:d} episodes! \tAverage Score: {:.2f}'.format(episode,
                                                                                      np.mean(rewards_on_100_episodes))) # end when average hits 200
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')  # end training
        break
